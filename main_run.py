from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr as psnr
import datagen
import model
import cv2
import pickle
import time
import tqdm
from operations import *

def main():

	phone_res=(100,100)
	camera_res=(100,100)
	num_images_per_step=100
	data_generator=datagen.DataGenerator('./dped/iphone/training_data','iphone',phone_res,camera_res,batch_size=30,training=True)
	main_model=model.WESPE(phone_res,camera_res)
	epochs=11
	test_generator=datagen.DataGenerator('./dped/iphone/test_data/patches','iphone',phone_res,camera_res)
	for j in range(epochs):
		print('Epoch_'+str(j+1)+'/'+str(epochs))
		data_generator.on_epoch_end()
		data_list=tqdm.tqdm(range(min(num_images_per_step,len(data_generator))))		
		for i in data_list:
			x,y,y_coupled=data_generator[i]
			losses,y_generated=main_model.train_step(x,y)
			losses['avg_psnr'],losses['avg_ssim']=compare_imgs(postprocess(y_generated),postprocess(y_coupled))
			data_list.set_description(str(losses))
			data_list.refresh()
			time.sleep(0.01)

		if j%2==0:
			for i in range(len(test_generator)):
				x,_=data_generator[i]
				cv2.imwrite('./org'+str(i)+'.jpg',postprocess(x[0]))
				img=main_model.predict(x).numpy()
				cv2.imwrite('./pred'+str(i)+'.jpg',postprocess(img[0]))
			with open('model.pkl','wb') as file:
				pickle.dump(main_model,file)		

	with open('model.pkl','wb') as file:
		pickle.dump(main_model,file)
	test_generator=datagen.DataGenerator('./dped/iphone/test_data','iphone',phone_res,camera_res)
	for i in range(len(test_generator)):
		x,_=data_generator[i]
		cv2.imwrite('./org'+str(i)+'.jpg',x[0]*255.0)
		img=main_model.predict(x).numpy()
		cv2.imwrite('./pred'+str(i)+'.jpg',img[0]*255.0)



def compare_imgs(generated_imgs, target_imgs):
    s_sim=0.0
    p_snr=0.0

    for i in range(len(generated_imgs)):
    	s_sim+=ssim(generated_imgs[i],target_imgs[i],multichannel=True)
    	p_snr+=psnr(target_imgs[i],generated_imgs[i])

    return p_snr/len(generated_imgs), s_sim/len(generated_imgs)



if __name__=='__main__':
	main()
