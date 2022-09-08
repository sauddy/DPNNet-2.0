
# DPNNet-2.0
Disk Planet Convolutional Neural Network

We introduce the architecture of DPNNet-2.0, second in the series after DPNNet \citep{aud20}, designed using a Convolutional Neural Network ( CNN, here specifically ResNet50) for predicting exoplanet masses directly from simulated images of protoplanetary disks hosting a single planet. DPNNet-2.0 additionally consists of a multi-input framework that uses both a CNN and multi-layer perceptron (a class of artificial neural network) for processing image and disk parameters simultaneously. This enables DPNNet-2.0 to be trained using images directly, with the added option of considering disk parameters (disk viscosities, disk temperatures, disk surface density profiles, dust abundances, and particle Stokes numbers) generated from disk-planet hydrodynamic simulations as inputs. This work provides the required framework and is the first step towards the use of computer vision (implementing CNN) to directly extract mass of an exoplanet from planetary gaps observed in dust-surface density maps by telescopes such as the Atacama Large (sub-)Millimeter Array.

The paper is accepted for publcation in Astrophysical Journal 
The arXiv version is available at https://arxiv.org/abs/2107.09086

![Disk_samples-white](https://user-images.githubusercontent.com/46558389/189010857-0513eeb8-0769-4ef6-aba2-b4521dbd0b9a.png)
![DPCNet_hybrid](https://user-images.githubusercontent.com/46558389/189010859-d339d326-10b0-4b26-920e-3c7935d956d1.png)
