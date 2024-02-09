# AsyncFedInv-Asynchronous-Federated-learning-for-Seismic-Inversion

In this project, we proposed the Asynchronous Federated Learning for Seismic Inversion (AsyncFedInv) framework, which applies multiple IoT devices in terms of edge computing boards to collaboratively train a compact UNet model in real-time based on a novel asynchronous federated learning, where 1) a staleness function is applied to mitigate model staleness, and 2) clients that generate similar local models would suspend its training, thus reducing the communication costs and energy consumption.

The overview of the whole AnsycFedInv architecture is shown in the image below:

![FL_Seismic.jpg](FL_Sesmic.jpg)

The dataset used in this work is the Salt Data which consist of the seismic data and their corresponding velocity models. The pairs are fed to the UNet model to the non-linear mapping from the seismic data to the velocity model. The seismic data specifications are $S\times R\times T$ = $29\times 301\times 201$ and velcity model specifications are $R\times T$ = $301\times 201$. The seismic and velocity model images are shown below:
![Seismic Image and Velocity Image](seis_pd.pdf)

We performed several simulations on the Salt data and the Simulation results demonstrate that AsyncFedInv achieves a similar convergence rate but lower training loss and better testing performance as compared to a baseline algorithm FedAvg
