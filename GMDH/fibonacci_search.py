
def check_fib(val: int, i: int, j: int) -> int:
    if j == val:
        return True
    else:
        check_fib(val, j, i + j)

