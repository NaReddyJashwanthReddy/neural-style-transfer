from torch import optim
from torchvision.models import vgg19 
import streamlit as st
from utils import LoadImage,Features,device
from Loss import NSTLoss
from logger import logging

logging.info("Started the project")

vgg=vgg19(pretrained=True).features
for params in vgg.parameters():
    params.requires_grad_(False)

logging.info("Extracted the model successfully")

context_image=st.file_uploader("Upload Your Content Image",type=['jpg','png','webp'])
logging.info("Successfully uploaded content Image")
style_image=st.file_uploader("Upload Your Style Image",type=['jpg','png','webp'])

logging.info("Successfully uploaded style Image")


if context_image is not None and style_image is not None and st.button("Start"):
    Image_Loader=LoadImage()
    context_image=Image_Loader.Load_Image(context_image)
    style_image=Image_Loader.Load_Image(style_image)

    logging.info("Image loading successfully Done")

    target_image=context_image.clone().requires_grad_(True).to(device)

    logging.info("target Image Loaded")

    context_layer=['21']
    style_layer=['0','5','10','19','28']

    style_weight = 1e7  
    content_weight = 1
    tv_weight = 1e-5
    learning_rate=0.003
    steps=100

    optimizer=optim.LBFGS([target_image],lr=learning_rate)

    logging.info("Done loading required things")

    place_holder=st.empty()
    step_holder=st.empty()
    logging.info("Start the training")

    for step in range(steps):
        feature=Features()
        def closure():
            logging.info("Get features started")
            target_features=feature.get_features(target_image,vgg,context_layer+style_layer)
            context_features=feature.get_features(context_image,vgg,context_layer)
            style_features=feature.get_features(style_image,vgg,style_layer)
            logging.info("Get features Ended")

            content_loss=0
            style_loss=0
            tv_loss=0
            Loss_function=NSTLoss()
            logging.info("Start calculate loss")
            for layer in context_layer:
                content_loss+=Loss_function.ContentLoss(context_features[layer],target_features[layer])
            logging.info(f"content loss calculated : {content_loss}")
            for layer in style_layer:
                style_loss+=Loss_function.StyleLoss(style_features[layer],target_features[layer])
            logging.info(f"styling loss calculated : {style_loss}")
            tv_loss=Loss_function.total_variable_loss(target_image)
            logging.info(f"tv loss calculated : {tv_loss}")

            loss=content_weight*content_loss+style_weight*style_loss+tv_weight*tv_loss
            logging.info(f"total loss : {loss}")
            optimizer.zero_grad()

            loss.backward()

            return loss

        optimizer.step(closure)

        clone_image=Image_Loader.Denormalize(target_image)
        clone_image=Image_Loader.ImShow(clone_image)
        logging.info("start showing image")
        place_holder.image(clone_image,caption="Combined Image")
        logging.info("end showing image")
        st.write(f"step : {step}")
        
        logging.info(f"epoch {step} done")
    st.write("### Download your Image Here")
    st.download_button(
        label='Download',
        data=target_image,
        file_name="OutputCombined.jpg"
    )
    logging.info("code completed")