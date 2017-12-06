from timeit import default_timer as timer
import pickle

start = timer()
with open('test', 'wb') as f:
	pickle.dump(5, f)
with open('test', 'rb') as f:
	a = pickle.load(f)
print(timer()-start, a)