import datagen
import model
import cv2
def main():

	phone_res=(100,100)
	camera_res=(100,100)
	data_generator=datagen.DataGenerator('./data/iphone','iphone',phone_res,camera_res)
	main_model=model.WESPE(phone_res,camera_res)

	for i in range(len(data_generator)-1):
		x,y=data_generator[i]
		print(main_model.train_step(x,y))

	x,_=data_generator[len(data_generator)]
	cv2.imwrite('./org.jpg',x[0]*255.0)
	img=main_model.predict(x).numpy()
	cv2.imwrite('./pred.jpg',img[0]*255.0)





if __name__=='__main__':
	main()