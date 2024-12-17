import multiprocessing
import time
from multiprocessing import Process

def func_print(x:int=2,y:int=4):
    import time
    time.sleep(1*60)
    print(x+y)

if __name__ == "__main__":
    #multiprocessing.set_start_method("spawn")
    process = multiprocessing.Process(target=func_print, args=(1, 3))
    process.start()
    process.join()
    