"""
GPU Stage Scheduler for RAG PDF Parser.

Manages GPU resource scheduling across processing stages to prevent OOM errors
and optimize throughput for batch processing.
"""

import asyncio
import time
from typing import List, Optional, Dict, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from queue import PriorityQueue
import threading


class ProcessingStage(Enum):
    """Processing pipeline stages."""
    LAYOUT = "layout"      # Surya layout detection
    OCR = "ocr"            # PaddleOCR text extraction
    VLM = "vlm"            # VLM captioning
    TRANSLATION = "translation"  # Translation


@dataclass
class StageConfig:
    """Configuration for a processing stage."""
    name: str
    priority: int                    # Lower = higher priority
    estimated_memory_mb: int         # Estimated GPU memory usage
    max_concurrent: int = 1          # Max concurrent tasks
    timeout_seconds: int = 300       # Task timeout


@dataclass(order=True)
class PageTask:
    """A page processing task."""
    priority: int
    page_num: int = field(compare=False)
    doc_id: str = field(compare=False)
    stage: ProcessingStage = field(compare=False)
    data: Any = field(compare=False, default=None)
    callback: Optional[Callable] = field(compare=False, default=None)


class GPUStageScheduler:
    """
    Manages GPU resource scheduling across processing stages.

    Stages are processed in order: Layout -> OCR -> VLM -> Translation
    Each stage can have concurrent limits to prevent GPU OOM.
    """

    DEFAULT_STAGES = {
        ProcessingStage.LAYOUT: StageConfig(
            name="Layout Detection",
            priority=1,
            estimated_memory_mb=2000,
            max_concurrent=1,
            timeout_seconds=120
        ),
        ProcessingStage.OCR: StageConfig(
            name="OCR Extraction",
            priority=2,
            estimated_memory_mb=1500,
            max_concurrent=2,
            timeout_seconds=180
        ),
        ProcessingStage.VLM: StageConfig(
            name="VLM Captioning",
            priority=3,
            estimated_memory_mb=4000,
            max_concurrent=3,
            timeout_seconds=300
        ),
        ProcessingStage.TRANSLATION: StageConfig(
            name="Translation",
            priority=4,
            estimated_memory_mb=2000,
            max_concurrent=2,
            timeout_seconds=120
        )
    }

    def __init__(
        self,
        max_gpu_memory_mb: int = 8000,
        stages: Optional[Dict[ProcessingStage, StageConfig]] = None
    ):
        """
        Initialize the GPU Stage Scheduler.

        Args:
            max_gpu_memory_mb: Maximum GPU memory available
            stages: Custom stage configurations
        """
        self.max_gpu_memory_mb = max_gpu_memory_mb
        self.stages = stages or self.DEFAULT_STAGES

        # Task queues per stage
        self._queues: Dict[ProcessingStage, PriorityQueue] = {
            stage: PriorityQueue() for stage in ProcessingStage
        }

        # Active task tracking
        self._active_tasks: Dict[ProcessingStage, int] = {
            stage: 0 for stage in ProcessingStage
        }

        # Semaphores for concurrency control
        self._semaphores: Dict[ProcessingStage, asyncio.Semaphore] = {}

        # Statistics
        self._stats = {
            'tasks_queued': 0,
            'tasks_completed': 0,
            'tasks_failed': 0,
            'total_wait_time': 0.0,
            'total_process_time': 0.0
        }

        self._lock = threading.Lock()
        self._running = False

    def _init_semaphores(self):
        """Initialize async semaphores (must be called in async context)."""
        for stage, config in self.stages.items():
            self._semaphores[stage] = asyncio.Semaphore(config.max_concurrent)

    async def schedule_task(
        self,
        task: PageTask,
        processor: Callable
    ) -> Any:
        """
        Schedule and execute a task with resource management.

        Args:
            task: PageTask to execute
            processor: Async callable that processes the task

        Returns:
            Result from the processor
        """
        if task.stage not in self._semaphores:
            self._init_semaphores()

        stage_config = self.stages[task.stage]
        semaphore = self._semaphores[task.stage]

        start_wait = time.time()

        async with semaphore:
            wait_time = time.time() - start_wait
            self._stats['total_wait_time'] += wait_time

            with self._lock:
                self._active_tasks[task.stage] += 1

            start_process = time.time()

            try:
                # Execute with timeout
                result = await asyncio.wait_for(
                    processor(task),
                    timeout=stage_config.timeout_seconds
                )
                self._stats['tasks_completed'] += 1
                return result

            except asyncio.TimeoutError:
                self._stats['tasks_failed'] += 1
                raise TimeoutError(
                    f"Task timed out after {stage_config.timeout_seconds}s: "
                    f"{task.stage.value} for page {task.page_num}"
                )

            except Exception as e:
                self._stats['tasks_failed'] += 1
                raise

            finally:
                process_time = time.time() - start_process
                self._stats['total_process_time'] += process_time

                with self._lock:
                    self._active_tasks[task.stage] -= 1

    async def process_page_pipeline(
        self,
        page_num: int,
        doc_id: str,
        page_data: Any,
        processors: Dict[ProcessingStage, Callable]
    ) -> Dict[ProcessingStage, Any]:
        """
        Process a page through the full pipeline.

        Args:
            page_num: Page number
            doc_id: Document ID
            page_data: Initial page data (e.g., PIL Image)
            processors: Dict of stage -> async processor function

        Returns:
            Dict of stage -> result
        """
        results = {}
        current_data = page_data

        for stage in [ProcessingStage.LAYOUT, ProcessingStage.OCR,
                      ProcessingStage.VLM, ProcessingStage.TRANSLATION]:

            if stage not in processors:
                continue

            task = PageTask(
                priority=self.stages[stage].priority,
                page_num=page_num,
                doc_id=doc_id,
                stage=stage,
                data=current_data
            )

            result = await self.schedule_task(task, processors[stage])
            results[stage] = result

            # Pass result to next stage
            current_data = result

        return results

    async def process_document_parallel(
        self,
        doc_id: str,
        pages: List[Any],
        processors: Dict[ProcessingStage, Callable],
        max_parallel_pages: int = 3
    ) -> List[Dict[ProcessingStage, Any]]:
        """
        Process multiple pages in parallel with stage-aware scheduling.

        Args:
            doc_id: Document ID
            pages: List of page data
            processors: Dict of stage -> async processor function
            max_parallel_pages: Max pages to process in parallel

        Returns:
            List of results per page
        """
        page_semaphore = asyncio.Semaphore(max_parallel_pages)

        async def process_single_page(page_num: int, page_data: Any):
            async with page_semaphore:
                return await self.process_page_pipeline(
                    page_num, doc_id, page_data, processors
                )

        tasks = [
            process_single_page(i + 1, page)
            for i, page in enumerate(pages)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"Page {i + 1} failed: {result}")
                processed_results.append({})
            else:
                processed_results.append(result)

        return processed_results

    def get_stats(self) -> Dict[str, Any]:
        """Get scheduler statistics."""
        with self._lock:
            stats = self._stats.copy()
            stats['active_tasks'] = dict(self._active_tasks)

            # Calculate averages
            if stats['tasks_completed'] > 0:
                stats['avg_wait_time'] = stats['total_wait_time'] / stats['tasks_completed']
                stats['avg_process_time'] = stats['total_process_time'] / stats['tasks_completed']
            else:
                stats['avg_wait_time'] = 0
                stats['avg_process_time'] = 0

            return stats

    def estimate_completion_time(
        self,
        remaining_pages: int,
        avg_page_time: Optional[float] = None
    ) -> float:
        """
        Estimate time to complete remaining pages.

        Args:
            remaining_pages: Number of pages remaining
            avg_page_time: Optional override for average page time

        Returns:
            Estimated seconds to completion
        """
        stats = self.get_stats()

        if avg_page_time is None:
            if stats['tasks_completed'] > 0:
                avg_page_time = stats['avg_process_time']
            else:
                # Default estimate: 30 seconds per page
                avg_page_time = 30.0

        # Account for parallelism
        min_concurrent = min(
            config.max_concurrent for config in self.stages.values()
        )

        return (remaining_pages / min_concurrent) * avg_page_time

    def reset_stats(self):
        """Reset scheduler statistics."""
        with self._lock:
            self._stats = {
                'tasks_queued': 0,
                'tasks_completed': 0,
                'tasks_failed': 0,
                'total_wait_time': 0.0,
                'total_process_time': 0.0
            }


# Convenience function for simple usage
async def schedule_page_processing(
    scheduler: GPUStageScheduler,
    page_num: int,
    doc_id: str,
    image,
    layout_fn,
    ocr_fn,
    vlm_fn
):
    """
    Convenience function to schedule page processing.

    Args:
        scheduler: GPUStageScheduler instance
        page_num: Page number
        doc_id: Document ID
        image: Page image
        layout_fn: Async layout detection function
        ocr_fn: Async OCR function
        vlm_fn: Async VLM caption function

    Returns:
        Dict of results by stage
    """
    processors = {
        ProcessingStage.LAYOUT: layout_fn,
        ProcessingStage.OCR: ocr_fn,
        ProcessingStage.VLM: vlm_fn
    }

    return await scheduler.process_page_pipeline(
        page_num, doc_id, image, processors
    )
