# ChauffeurNet:
ChauffeurNet : Learning to Drive by Imitating the Best and Synthesizing the Worst.
Reproduction the result according to this paper[https://arxiv.org/pdf/1812.03079.pdf].
I just implement it on the basis of my comprehension，because the paper didn't introduce the neural network in every detail.
The model is implemented by Keras with Tensorflow backend.

# Roadmap:
1.Model and train and prediction with mocked data.[done]
2.Data pipeline for real data.
3.Train it in real world data.
4.Other approachs in paper.
5.Test it in simulation.
  I want the model can be used in different simulation environment.
  Welcome other contributors to integrate different open source or private simulators. 
  I will combine my company's simulator and some simple simulators first.
6.Test it in Real world on china's urban road.

# Model options:
1.use conv layers like U-Net(Conv+Upsampling/Deconv) [done]
2.Conv + Full Connect like artari-net
3.Fully Conv
4.Fully Conv + GRU

# Links:
https://github.com/Iftimie/ChauffeurNet

# Install:
anaconda3, python 3.6, keras 2.2.4, tensorflow 1.12.0
