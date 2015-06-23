import pygame.camera
import pygame.image

def tirafoto():
	pygame.camera.init()
	cam = pygame.camera.Camera(pygame.camera.list_cameras()[0])
	cam.start()
	
	i=1
	while i<=1:
		img = cam.get_image()
		newimage =  (("photo"+"%s"+".jpg") % str(i)) 
		newimage = join('./agora/', newimage)
		pygame.image.save(img, newimage)
		sleep(1)
		i+=1
		listatifafoto.append(img)
	pygame.camera.quit()
tirafoto()