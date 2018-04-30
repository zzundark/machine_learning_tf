import config as cfg
from inputs import input_pipeline 
from solver import solver 
from softmax_regression import softmax_regression


def main():
	input=input_pipeline()
	print("ready input pipeline")
	net=softmax_regression()
	_solver=solver(net,input,'./log')
	_solver.train_and_test()
	
	


if __name__ == '__main__':
	main()