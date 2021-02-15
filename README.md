# Generative-Models

## Generative-Adversarial-Networks (GAN)

#### MNIST 데이터 셋 
   
 - [GAN](https://github.com/sujiny-tech/Generative-Adversarial-Networks/blob/main/MNIST/GAN_MNIST.py)
    + epoch = 10000   
      <img src="https://user-images.githubusercontent.com/72974863/101121901-8dcaa780-3634-11eb-949a-9a4765a32794.png" width="40%" hegiht="40%">   
    + epoch = 100000   
      <img src="https://user-images.githubusercontent.com/72974863/101121950-aa66df80-3634-11eb-9cc8-a794100813f9.png" width="40%" hegiht="40%"> 
    
 - [LSGAN](https://github.com/sujiny-tech/Generative-Adversarial-Networks/blob/main/MNIST/LSGAN_MNIST.py)
    + epoch = 10000   
      <img src="https://user-images.githubusercontent.com/72974863/101122768-8f956a80-3636-11eb-9e3b-d27bca00ddc7.png" width="40%" height="40%">
    + epoch = 100000   
      <img src="https://user-images.githubusercontent.com/72974863/101122756-899f8980-3636-11eb-83e0-522746058520.png" width="40%" height="40%">

 - [DCGAN](https://github.com/sujiny-tech/Generative-Adversarial-Networks/blob/main/MNIST/DCGAN_MNIST.py)
    + epoch = 10000   
      <img src="https://user-images.githubusercontent.com/72974863/101122862-d71bf680-3636-11eb-83d7-d9d2313b61f6.png" width="40%" height="40%">
    + epoch = 100000   
      <img src="https://user-images.githubusercontent.com/72974863/101122875-dbe0aa80-3636-11eb-9336-d002b9604a4e.png" width="40%" height="40%">

 - [WGAN](https://github.com/sujiny-tech/Generative-Adversarial-Networks/blob/main/MNIST/WGAN_MNIST.py)
    + epoch = 10000   
      <img src="https://user-images.githubusercontent.com/72974863/101122964-10ecfd00-3637-11eb-87e3-f4b0991fcaa2.png" width="40%" height="40%">
    + epoch = 100000   
      <img src="https://user-images.githubusercontent.com/72974863/101122979-164a4780-3637-11eb-85ad-519fb8ff69c8.png" width="40%" height="40%">

 - [CGAN](https://github.com/sujiny-tech/Generative-Adversarial-Networks/blob/main/MNIST/CGAN_MNIST.py)
    + epoch = 10000 (숫자 4에 대한 synthetic data)   
      <img src="https://user-images.githubusercontent.com/72974863/101123142-793bde80-3637-11eb-9627-9c0065f6d1e6.png" width="40%" height="40%">

    + epoch = 100000 (숫자 4에 대한 synthetic data)   
      <img src="https://user-images.githubusercontent.com/72974863/101123112-6923ff00-3637-11eb-9b20-7ba6fab2293a.png" width="40%" height="40%">

 - [InfoGAN](https://github.com/sujiny-tech/Generative-Adversarial-Networks/blob/main/MNIST/InfoGAN_MNIST.py)
    + epoch = 10000   
      <img src="https://user-images.githubusercontent.com/72974863/101123292-df286600-3637-11eb-8a85-6ecf1b18f67b.png" width="40%" hegiht="40%">
    + epoch = 100000   
      <img src="https://user-images.githubusercontent.com/72974863/101123296-e3ed1a00-3637-11eb-84fd-1393770cb429.png" width="40%" height="40%">

 - [cycleGAN](https://github.com/sujiny-tech/Generative-Adversarial-Networks/blob/main/MNIST/cycleGAN_MNIST_svhn_model_v2.py) : 파라미터 조절 필요

 
#### Celeba 데이터 셋
   
 - DCGAN 
    + [RGB, 28x28 픽셀 데이터](https://github.com/sujiny-tech/Generative-Adversarial-Networks/blob/main/Celeba/DCGAN_celeba.py)
       + batch size 128, epoch = 200   
         <img src="https://user-images.githubusercontent.com/72974863/101123537-6b3a8d80-3638-11eb-9aa2-0eece6d2d76b.png" width="40%" height="40%">
       + batch size 500, epoch = 200   
         <img src="https://user-images.githubusercontent.com/72974863/101124100-b2754e00-3639-11eb-9748-6497c212b1ad.png" width="40%" height="40%">

    + [RGB, 84x84 픽셀 데이터](https://github.com/sujiny-tech/Generative-Adversarial-Networks/blob/main/Celeba/DCGAN_celeba_84size.py)
       + epoch = 110   
         <img src="https://user-images.githubusercontent.com/72974863/101123995-688c6800-3639-11eb-9213-b782c566ffd2.png" width="40%" hegiht="40%">
         
    + [2번 버전(RGB, 84x84 픽셀 데이터)의 배치사이즈 변경한 버전](https://github.com/sujiny-tech/Generative-Adversarial-Networks/blob/main/Celeba/origin2_ver2_batchsize.py)
       + epoch =110   
         <img src="https://user-images.githubusercontent.com/72974863/101123932-41ce3180-3639-11eb-8b6f-33351d73bbc4.png" width="40%" height="40%">
    + [RGB, 28x28 픽셀 데이터에서 학습된 모델을 기반으로 vector arithmetic 수행]
       + [DCGAN_latent_vector_1](https://github.com/sujiny-tech/Generative-Adversarial-Networks/blob/main/Celeba/DCGAN_latent_vector.py)  
         <img src="https://user-images.githubusercontent.com/72974863/101124750-f74db480-363a-11eb-9294-5c691468c3f2.png" width="40%" height="40%">

       + [vector_arithmetic_2](https://github.com/sujiny-tech/Generative-Adversarial-Networks/blob/main/Celeba/vector_arithmetic.py)   
          + smiling_woman(1행 이미지) - neutral_woman(2행 이미지) + neutral_man(3행 이미지) = smiling_man ?   
            <img src="https://user-images.githubusercontent.com/72974863/101124957-675c3a80-363b-11eb-94d2-ed90c2d36e7d.png" width="40%" height="40%">
          + smiling_man !   
            <img src="https://user-images.githubusercontent.com/72974863/101125101-c15d0000-363b-11eb-9191-b585ff4f9d3d.png" width="40%" height="40%">
          + glasses_man(1행 이미지) - neutral_man(2행 이미지) + neutral_woman(3행 이미지) = glasses_woman ?   
            <img src="https://user-images.githubusercontent.com/72974863/101125374-71326d80-363c-11eb-91f7-14d08d884e01.png" width="40%" height="40%">
          + glasses_woman ! ~~(3번째 안경...? 🙄)~~   
            <img src="https://user-images.githubusercontent.com/72974863/101125422-8dcea580-363c-11eb-8ea1-9694182df39f.png" width="40%" height="40%">
       
 - [cycleGAN](https://github.com/sujiny-tech/Generative-Adversarial-Networks/blob/main/Celeba/cycleGAN_Celeba.py) : 파라미터 조절 필요


#### Fashion-MNIST 데이터 셋

 - CGAN : 
    + [fully connected layer 구조](https://github.com/sujiny-tech/Generative-Adversarial-Networks/blob/main/Fashion-MNIST/CGAN_dense_layer1.py)
       + epoch =1000   
         <img src="https://user-images.githubusercontent.com/72974863/101125909-b4411080-363d-11eb-9841-ac193d278e8d.png" width="40%" height="40%">   
         <img src="https://user-images.githubusercontent.com/72974863/101125987-de92ce00-363d-11eb-87f7-4e05672afd9e.png" width="40%" height="40%">
         
    + [deep convolutional layer 구조_1](https://github.com/sujiny-tech/Generative-Adversarial-Networks/blob/main/Fashion-MNIST/CGAN_Fashion_MNIST.py)
    + [deep convolutional layer 구조_2](https://github.com/sujiny-tech/Generative-Adversarial-Networks/blob/main/Fashion-MNIST/CGAN_structure_change.py)
       + epoch =100   
         <img src="https://user-images.githubusercontent.com/72974863/101126667-50b7e280-363f-11eb-8877-42565c2ac69c.png" width="40%" height="40%">   
         
 - [WGAN](https://github.com/sujiny-tech/Generative-Adversarial-Networks/blob/main/Fashion-MNIST/WGAN_Fashion_MNIST.py)
    + epoch = 9800 (이 모델로 mnist 데이터셋)   
      <img src="https://user-images.githubusercontent.com/72974863/101126968-e489ae80-363f-11eb-9f34-f7db7ec40a03.png" width="40%" height="40%">
    + epoch = 10000 (fashion-mnist 데이터셋)   
      <img src="https://user-images.githubusercontent.com/72974863/101127199-5d890600-3640-11eb-9976-2b21f56859ee.png" width="40%" height="40%">
    
 - [WGAN-GP](https://github.com/sujiny-tech/Generative-Adversarial-Networks/blob/main/Fashion-MNIST/WGAN_Fashion_MNIST.py)
    + epoch = 400 (이 모델로 mnist 데이터셋)   
      <img src="https://user-images.githubusercontent.com/72974863/101127154-42b69180-3640-11eb-95b7-69cde1107000.png" width="40%" height="40%">
    + epoch = 10000 (fashion-mnist 데이터셋)   
      <img src="https://user-images.githubusercontent.com/72974863/101127265-8d380e00-3640-11eb-9d31-99e7595dd5b2.png" width="40%" height="40%">

---------------

## Restricted Boltzmann Machine(RBM)

#### MNIST 데이터 셋
 - [Binary_RBM](https://github.com/sujiny-tech/Generative-Models/blob/main/Restricted%20Boltzmann%20Machine(RBM)/Binary_RBM_MNIST.py)
    + CD-1
       <img src="https://user-images.githubusercontent.com/72974863/107949512-ba0d8780-6fd8-11eb-92f0-476228a42867.png" width="40%" height="40%">    
       
       
    + CD-2
       <img src="https://user-images.githubusercontent.com/72974863/107949530-c2fe5900-6fd8-11eb-9c2f-3cc849734d40.png" width="40%" height="40%">    
    
    + CD-20
       <img src="https://user-images.githubusercontent.com/72974863/107949550-c98cd080-6fd8-11eb-9571-47efd04c7b4f.png" width="40%" height="40%">    
    
    + CD-50
       <img src="(https://user-images.githubusercontent.com/72974863/107949570-cf82b180-6fd8-11eb-929e-ede1e5761ee8.png" width="40%" height="40%">    
       
