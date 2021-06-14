# FacadeStream
Visualization tool for Facade Segmentation



# TL;DR

**Installing Dependencies**
In your virtual environment (for example, anaconda), install all dependencies by running: 

```
pip3 install -r requirements.txt
```



**Running the Streamlit App**

```
streamlit run scanner.py
```



**Then download the pretrained model by clicking the button**. 



---



# How to Run the App



## Step 0: Virutal Environment [Optional]

It is convenient to use pip to manage python environment. 

To check the current pip version: 

```shell
pip3 --version
```

Installing the latest pip:

```
pip3 install --upgrade pip
```



### Installing virtualenv

**virtualenv:** https://pypi.org/project/virtualenv/

```
pip3 install virtualenv
```



### Creating a virtual environment,  'facade-env' (or any other names)

``` 
virtualenv facade-env
```



### Activating the virtual environment

**Windows**

```
facade-env\scripts\activate
```

**Mac & Linux**

```
source facade-env/bin/activate
```



### Deactivating the virtual environment

To leave the virtual environment when everything is done (the localhost is running, in our case). 

```
deactivate
```



## Step 1: 'Pip Installs Packages'

Change directory to the folder. In your pip environment, install the dependencies using:

```
pip3 install -r requirements.txt
```



## Step 2: Import Trained Model

**Download the pretrained model by clicking the button.**

or 

Download the trained model using following link: 

https://drive.google.com/uc?export=download&id=1rJ3edeARtcprrgs14lj5iZLTLkn9kufw

, and put it under **/models** folder.



## Step 3: Run the scanner app

**Streamlit:** https://streamlit.io

Start the facade segmentation tool by running:

```
streamlit run scanner.py
```



## Others

### Color Map

Using eTRIMES's color code

More about **eTRIMS dataset**: http://www.ipb.uni-bonn.de/projects/etrims_db/

| Index | Label      | RGB             |
| --- | --------- | --------------- |
|   0   | Various    | (0, 0, 0)       |
|   1   | Wall       | (128, 0, 0)     |
|   2   | Car        | (128, 0, 128)   |
|   3   | Door       | (128, 128, 0)   |
|   4   | Pavement   | (128, 128, 128) |
|   5   | Road       | (128, 64, 0)    |
|   6   | Sky        | (0, 128, 128)   |
|   7   | Vegetation | (0, 128, 0)     |
|   8   | Window     | (0, 0, 128)     |

