## MongoDB Database Creation

### Step-1: Login to Mongo DB

*  Go to [Mongo DB](https://www.mongodb.com/)

* Login with your GMAIL credentials

### Step-2: Create a new Cluster

* Click on **Create a Cluster**

<p align="left">
<img src="images/1.png" width="1080" height="480">
</p>

* Select **Free Plan** 

<p align="left">
<img src="images/2.png" width="1080" height="480">
</p>

* Give a database name **"ChatBot"** in **Configurations**

* Click on **Create Deployment**

### Step-3: Go to Database & Network Access

<p align="left">
<img src="images/3.png" width="1080" height="480">
</p>

* Add a **New Database User**

<p align="left">
<img src="images/4.png" width="1080" height="480">
</p>

* Add **password authentication**

<p align="left">
<img src="images/5.png" width="1080" height="480">
</p>

* Select **Atlas admin**  as **Built-in Role** and select **Add user**

<p align="left">
<img src="images/6.png" width="1080" height="480">
</p>

* New database user is added

<p align="left">
<img src="images/7.png" width="1080" height="480">
</p>

### Step-4: Add IP address

* Go to **NETWORK ACCESS -> IP Access List -> Add IP Address**

<p align="left">
<img src="images/8.png" width="1080" height="480">
</p>

### Step-5: Connect to ChatBot

* Go to **DATABASE -> Clusters -> ChatBot (Connect)**

<p align="left">
<img src="images/9.png" width="1080" height="480">
</p>

* Select **Drivers** and add the connection string to .env file 

<p align="left">
<img src="images/10.png" width="1080" height="480">
</p>
