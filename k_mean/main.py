import config as cfg
from inputs import input_pipeline 
from solver import solver
from k_mean import k_mean


def main():
	input=input_pipeline()
	print("ready input pipeline")
	net=k_mean()
	_solver=solver(net,input,'./log')
	_solver.train_and_test()
	
	


if __name__ == '__main__':
	main()