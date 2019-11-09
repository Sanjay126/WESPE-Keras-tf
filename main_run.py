import datagen
import model

def main():

	phone_res=(100,100)
	camera_res=(100,100)
	data_generator=datagen.DataGenerator('./drive/My Drive/data/iphone','iphone',phone_res,camera_res)
	main_model=model.WESPE(phone_res,camera_res)

	for i in range(len(data_generator)):
		x,y=data_generator[i]
		print(main_model.train_step(x,y))







if __name__=='__main__':
	main()