import re

def convert_cgpa(val):
    if isinstance(val, str) and "-" in val:
        low, high = val.split("-")
        return (float(low) + float(high)) / 2
    return float(val)

def convert_sleep(val):
    if isinstance(val, str):
        val = val.lower().replace("hrs", "").strip()
        if "-" in val:
            low, high = val.split("-")
            return (float(low) + float(high)) / 2
        return float(val)
    return float(val)

def convert_sports(val):
    if not isinstance(val, str):
        return val
    val = val.lower().strip()
    if "no" in val:
        return 0
    numbers = re.findall(r"\d+", val)
    if len(numbers) == 1:
        return float(numbers[0])
    if len(numbers) >= 2:
        nums = list(map(float, numbers))
        return sum(nums) / len(nums)
    return 0

def stress_activity_binary(val):
    return 1 if isinstance(val, str) and val.strip() != "" else 0

def convert_campus_discrimination(val):
    if isinstance(val, str):
        return 1 if val.strip().lower() == "yes" else 0
    return 0
