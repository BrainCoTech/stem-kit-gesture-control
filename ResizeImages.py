from PIL import Image

total_num_train_images = 1000
total_num_test_images = 100

def resizeImage(imageName):
    basewidth = 100
    img = Image.open(imageName)
    wpercent = (basewidth/float(img.size[0]))
    hsize = int((float(img.size[1])*float(wpercent)))
    img = img.resize((basewidth,hsize), Image.ANTIALIAS)
    img.save(imageName)

for i in range(0, total_num_train_images):
    # Mention the directory in which you wanna resize the images followed by the image name

    
    resizeImage("Dataset/PaperImages/paper_" + str(i) + '.png')
    resizeImage("Dataset/RockImages/rock_" + str(i) + '.png')
    resizeImage("Dataset/ScissorImages/scissor_" + str(i) + '.png')




for i in range(0, total_num_test_images):
    resizeImage("Dataset/PaperTest/paper_" + str(i) + '.png')
    resizeImage("Dataset/RockTest/rock_" + str(i) + '.png')
    resizeImage("Dataset/ScissorTest/scissor_" + str(i) + '.png')
