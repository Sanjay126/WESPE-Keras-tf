import datagen
import model

def main():

	phone_res=(100,100)
	camera_res=(100,100)
	data_generator=datagen.DataGenerator('./data/iphone','iphone',phone_res,camera_res)
	main_model=model.WESPE(phone_res,camera_res)

	for x,y in data_generator:
		print(main_model.train(x,y))







if __name__=='__main__':
	main()