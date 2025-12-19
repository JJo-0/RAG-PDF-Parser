import sys
import os

print("Python Executable:", sys.executable)
print("System Path:")
for p in sys.path:
    print(p)

print("\nAttempting imports...")
try:
    import surya
    print(f"Success: surya imported from {surya.__file__}")
except ImportError as e:
    print(f"Fail: surya ({e})")

try:
    import paddleocr
    print(f"Success: paddleocr imported from {paddleocr.__file__}")
except ImportError as e:
    print(f"Fail: paddleocr ({e})")
