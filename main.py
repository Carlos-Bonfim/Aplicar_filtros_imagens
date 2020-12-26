# import bibliotecas
from typing import List

import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageEnhance
from io import BytesIO
import base64

st.set_option('deprecation.showfileUploaderEncoding', False)


def get_image_download_link(img):
    # Generates a link allowing the PIL image to be downloaded
    # in:  PIL image
    # out: href string
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/jpg;base64,{img_str}" download="My_photo.jpg">Download result</a>'
    return href


def main():
    banner = Image.open('images/banner.png')
    st.sidebar.image(banner, width=300)
    st.sidebar.subheader("Meu Portfolio, clique abaixo:")
    link = '[*carlosbonfim.com*](https://carlosbonfim.com)'
    st.sidebar.markdown(link, unsafe_allow_html=True)

    st.title('**Projeto 1 - Aplicação de filtros em Imagens**')
    st.sidebar.title('**Filtros**')

    # Menu com opções diferentes de páginas
    #opcoes_menu = ['Filtros', 'Sobre']
    #st.sidebar.selectbox('Escolha uma opção', opcoes_menu)
    load_image = Image.open('images/empty.jpg')

    # carregando uma imagem
    type_images = ['jpeg', 'jpg', 'png']
    image_file = st.file_uploader('Carregue uma foto e aplique o filtro no menu lateral', type=type_images)

    if image_file is not None:
        load_image = Image.open(image_file)
        st.sidebar.text('Imagem Original')
        st.sidebar.image(load_image, width=100)


    # filtros à serem aplicados
    filtros = st.sidebar.selectbox('**Selecione um filtro**', ['Original', 'Grayscale', 'Sketch', 'Sépia', 'Blur', 'Canny',
                                               'Contraste e brilho', 'Sharpness'])

    if filtros == 'Grayscale':
        converted_image = np.array(load_image.convert('RGB'))
        gray_image = cv2.cvtColor(converted_image, cv2.COLOR_RGB2GRAY)

        opt_size = st.sidebar.slider('Selecione o tamanho', 1, 200, 50)

        width = int(load_image.size[0] * opt_size / 100)
        height = int(load_image.size[1] * opt_size / 100)
        dim = (width, height)
        
        st.sidebar.header("Ajuste o recorte da foto")
        left = st.sidebar.slider('Ajuste superior', 1, int(load_image.size[1]), 0)
        upper = st.sidebar.slider('Ajuste à esquerda', 1, int(load_image.size[1]), 0)
        right = st.sidebar.slider('Ajuste inferior', 1, int(load_image.size[0]), int(load_image.size[0]))
        lower = st.sidebar.slider('Ajuste à direita', 1, int(load_image.size[0]), int(load_image.size[0]))

        edited_photo_cont_gray = cv2.resize(gray_image[left:(right+1), upper:(lower+1)], dim, interpolation=cv2.INTER_AREA)

        st.image(edited_photo_cont_gray)

        # download da imagem
        result = Image.fromarray(edited_photo_cont_gray)
        st.markdown(get_image_download_link(result), unsafe_allow_html=True)

    elif filtros == 'Sketch':
        sketch_amount = st.sidebar.slider('Selecione a intensidade de contorno', 3, 81, 9, step=2)
        converted_image = np.array(load_image.convert('RGB'))
        gray_image = cv2.cvtColor(converted_image, cv2.COLOR_RGB2GRAY)
        inv_gray_image = 255 - gray_image
        blur_image = cv2.GaussianBlur(inv_gray_image, (sketch_amount, sketch_amount), 0, 0)
        sketch_image = cv2.divide(gray_image, 255 - blur_image, scale=256)

        opt_size = st.sidebar.slider('Selecione o tamanho', 1, 200, 50)

        width = int(load_image.size[0] * opt_size / 100)
        height = int(load_image.size[1] * opt_size / 100)
        dim = (width, height)

        st.sidebar.header("Ajuste o recorte da foto")
        left = st.sidebar.slider('Ajuste superior', 1, int(load_image.size[1]), 0)
        upper = st.sidebar.slider('Ajuste à esquerda', 1, int(load_image.size[1]), 0)
        right = st.sidebar.slider('Ajuste inferior', 1, int(load_image.size[0]), int(load_image.size[0]))
        lower = st.sidebar.slider('Ajuste à direita', 1, int(load_image.size[0]), int(load_image.size[0]))

        edited_photo_cont_sketch = cv2.resize(sketch_image[left:(right+1), upper:(lower+1)], dim, interpolation=cv2.INTER_AREA)

        st.image(edited_photo_cont_sketch)

        # download da imagem
        result = Image.fromarray(edited_photo_cont_sketch)
        st.markdown(get_image_download_link(result), unsafe_allow_html=True)

    elif filtros == 'Sépia':
        sepia_amount = st.sidebar.slider('Selecione a intensidade', 1, 3, 1, step=1)
        converted_image = np.array(load_image.convert('RGB'))
        kernel = np.array([[0.272, 0.534, 0.131],
                           [0.349, 0.686, 0.168],
                           [0.393, 0.769, 0.189]])
        teste = kernel * sepia_amount
        sepia_image = cv2.filter2D(converted_image, -1, teste)

        opt_size = st.sidebar.slider('Selecione o tamanho', 1, 200, 50)

        width = int(load_image.size[0] * opt_size / 100)
        height = int(load_image.size[1] * opt_size / 100)
        dim = (width, height)

        st.sidebar.header("Ajuste o recorte da foto")
        left = st.sidebar.slider('Ajuste superior', 1, int(load_image.size[1]), 0)
        upper = st.sidebar.slider('Ajuste à esquerda', 1, int(load_image.size[1]), 0)
        right = st.sidebar.slider('Ajuste inferior', 1, int(load_image.size[0]), int(load_image.size[0]))
        lower = st.sidebar.slider('Ajuste à direita', 1, int(load_image.size[0]), int(load_image.size[0]))

        edited_photo_cont_sepia = cv2.resize(sepia_image[left:(right+1), upper:(lower+1)], dim, interpolation=cv2.INTER_AREA)

        #st.image(edited_photo_cont_sepia, channels='BGR')
        st.image(edited_photo_cont_sepia, channels='RGB')

        # download da imagem
        result = Image.fromarray(edited_photo_cont_sepia)
        st.markdown(get_image_download_link(result), unsafe_allow_html=True)


    elif filtros == 'Blur':
        b_amount = st.sidebar.slider('Selecione a intensidade', 3, 81, 9, step=2)
        converted_image = np.array(load_image.convert('RGB'))
        converted_image = cv2.cvtColor(converted_image, cv2.COLOR_RGB2BGR)
        blur_image = cv2.GaussianBlur(converted_image, (b_amount, b_amount), 0, 0)

        opt_size = st.sidebar.slider('Selecione o tamanho', 1, 200, 50)

        width = int(load_image.size[0] * opt_size / 100)
        height = int(load_image.size[1] * opt_size / 100)
        dim = (width, height)

        st.sidebar.header("Ajuste o recorte da foto")
        left = st.sidebar.slider('Ajuste superior', 1, int(load_image.size[1]), 0)
        upper = st.sidebar.slider('Ajuste à esquerda', 1, int(load_image.size[1]), 0)
        right = st.sidebar.slider('Ajuste inferior', 1, int(load_image.size[0]), int(load_image.size[0]))
        lower = st.sidebar.slider('Ajuste à direita', 1, int(load_image.size[0]), int(load_image.size[0]))

        edited_photo_cont_blur = cv2.resize(blur_image[left:(right+1), upper:(lower+1)], dim, interpolation=cv2.INTER_AREA)

        st.image(edited_photo_cont_blur, channels='BGR')

        # download da imagem
        result = Image.fromarray(edited_photo_cont_blur)
        st.markdown(get_image_download_link(result), unsafe_allow_html=True)

    elif filtros == 'Canny':
        b_amount = st.sidebar.slider('Selecione a intensidade', 1, 33, 3, step=2)
        converted_image = np.array(load_image.convert('RGB'))
        converted_image = cv2.cvtColor(converted_image, cv2.COLOR_RGB2BGR)
        blur_image = cv2.GaussianBlur(converted_image, (b_amount, b_amount), 0, 0)
        canny = cv2.Canny(blur_image, 100, 150)

        opt_size = st.sidebar.slider('Selecione o tamanho', 1, 200, 50)

        width = int(load_image.size[0] * opt_size / 100)
        height = int(load_image.size[1] * opt_size / 100)
        dim = (width, height)

        st.sidebar.header("Ajuste o recorte da foto")
        left = st.sidebar.slider('Ajuste superior', 1, int(load_image.size[1]), 0)
        upper = st.sidebar.slider('Ajuste à esquerda', 1, int(load_image.size[1]), 0)
        right = st.sidebar.slider('Ajuste inferior', 1, int(load_image.size[0]), int(load_image.size[0]))
        lower = st.sidebar.slider('Ajuste à direita', 1, int(load_image.size[0]), int(load_image.size[0]))

        edited_photo_cont_canny = cv2.resize(canny[left:(right+1), upper:(lower+1)], dim, interpolation=cv2.INTER_AREA)
        st.image(edited_photo_cont_canny)

        # download da imagem
        result = Image.fromarray(edited_photo_cont_canny)
        st.markdown(get_image_download_link(result), unsafe_allow_html=True)

    elif filtros == 'Contraste e brilho':
        bb_amount = st.sidebar.slider('brilho', -100, 100, 0)
        cc_amount = st.sidebar.slider('Contraste', -100, 100, 0)
        brightness = bb_amount
        contrast = cc_amount
        img = np.int16(load_image)
        img = img * (contrast / 127 + 1) - contrast + brightness
        img = np.clip(img, 0, 255)
        img = np.uint8(img)

        opt_size = st.sidebar.slider('Selecione o tamanho', 1, 200, 50)

        width = int(load_image.size[0] * opt_size / 100)
        height = int(load_image.size[1] * opt_size / 100)
        dim = (width, height)

        st.sidebar.header("Ajuste o recorte da foto")
        left = st.sidebar.slider('Ajuste superior', 1, int(load_image.size[1]), 0)
        upper = st.sidebar.slider('Ajuste à esquerda', 1, int(load_image.size[1]), 0)
        right = st.sidebar.slider('Ajuste inferior', 1, int(load_image.size[0]), int(load_image.size[0]))
        lower = st.sidebar.slider('Ajuste à direita', 1, int(load_image.size[0]), int(load_image.size[0]))

        edited_photo_cont_bril = cv2.resize(img[left:(right+1), upper:(lower+1)], dim, interpolation=cv2.INTER_AREA)
        st.image(edited_photo_cont_bril)

        # download da imagem
        result = Image.fromarray(edited_photo_cont_bril)
        st.markdown(get_image_download_link(result), unsafe_allow_html=True)


    elif filtros == 'Sharpness':
        sharp_amount = st.sidebar.slider('Selecione a intensidade', 0, 10, 1, step=1)
        converted_image = np.array(load_image.convert('RGB'))
        #converted_image = cv2.cvtColor(converted_image, cv2.COLOR_RGB2BGR)
        kernel = np.array([[-1.0, -1.0, -1.0],
                           [-1.0, 9.0, -1.0],
                           [-1.0, -1.0, -1.0]])
        teste = kernel * sharp_amount
        sharp_image = cv2.filter2D(converted_image, -1, teste)

        opt_size = st.sidebar.slider('Selecione o tamanho', 1, 200, 50)

        width = int(load_image.size[0] * opt_size / 100)
        height = int(load_image.size[1] * opt_size / 100)
        dim = (width, height)

        st.sidebar.header("Ajuste o recorte da foto")
        left = st.sidebar.slider('Ajuste superior', 1, int(load_image.size[1]), 0)
        upper = st.sidebar.slider('Ajuste à esquerda', 1, int(load_image.size[1]), 0)
        right = st.sidebar.slider('Ajuste inferior', 1, int(load_image.size[0]), int(load_image.size[0]))
        lower = st.sidebar.slider('Ajuste à direita', 1, int(load_image.size[0]), int(load_image.size[0]))


        edited_photo_sharp = cv2.resize(sharp_image[left:(right + 1), upper:(lower + 1)], dim, interpolation=cv2.INTER_AREA)

        st.image(edited_photo_sharp, channels='RGB')

        # download da imagem
        result = Image.fromarray(edited_photo_sharp)
        st.markdown(get_image_download_link(result), unsafe_allow_html=True)


    elif filtros == 'Original':
        opt_size = st.sidebar.slider('Selecione o tamanho', min_value=int(load_image.size[0]/2), max_value=load_image.size[0])
        st.image(load_image, width=opt_size)

    else:
        opt_size = st.sidebar.slider('Selecione o tamanho', min_value=int(load_image.size[0]/2), max_value=load_image.size[0])
        st.image(load_image, width=opt_size)




if __name__ == '__main__':
    main()