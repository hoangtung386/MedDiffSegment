import subprocess
try:
    result = subprocess.run(["python3", "scripts/train.py", "--help"], capture_output=True, text=True, check=True)
    print("train.py is executable and imports work.")
except subprocess.CalledProcessError as e:
    print("train.py failed:")
    print(e.stderr)
