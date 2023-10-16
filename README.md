# Xtern24

Ridge Falco's work sample for Xtern Artificial Intelligence Work Prompt

# Exploratory Data Analysis

When looking at the data set I first wanted to get the counts/underlying distribution for the features.

Here is the quick graph I made for Year: 

![image](https://github.com/RidgeFalco/Xtern24/assets/89974909/311c4166-e071-43d8-b147-f71a364fd7b2)

What we can notice from these counts is that most people who are ordering are in their second or third year. We can essentially think of first year or last year orders as outliers. This can be usful to know as a buisness as we know have a better idea for who are customers are. We could use this information to either make sure second and third year students choose use, or we could try to start a new marketing campaign to entice first year and last year students. 

For Major:

![image](https://github.com/RidgeFalco/Xtern24/assets/89974909/d5da2d70-b0e3-40fc-9110-f46b9974dcb2)

Here we can notice that STEM majors seem to be ordering here a lot more than non-STEM majors and specifically Math, Bio, Chem, Physics, Econ, and Astronomy majors describe the majority of the people who order. Using this information we could either try to double down on being STEM oriented or nuturing our STEM customer base, or try to branch out and get more non-STEM major customers.

For University:

![image](https://github.com/RidgeFalco/Xtern24/assets/89974909/b042eaaf-6bd5-49c8-bc89-9c26c5575e47)

When looking at what university our customers are from we see that most are from ISU, Ball State, Butler, and IU. This is incredibly important to know as we know that if we spend more resources at one of these campuses, there will be customers who will be willing to pay. For example, maybe a food truck is looking to operate on only one campus, well now they know that they should choose between ISU, Ball State, Butler, and IU since that is where most of their customer base already. This information might also be important to know for logistics and ingredient procurement.

Finally, for Time (Couldn't get the exact counts for some reason):

![image](https://github.com/RidgeFalco/Xtern24/assets/89974909/82cbf57d-9e32-476c-8ad2-22c0fd855be6)

Here we can notice that most of the order occur from 10 AM - 4 PM. This is obviously important to know as a food truck might want to shut down early to save on operations cost, especially if they know that only a handful of orders will come in. It is also useful for knowing how many people to staff and when they should be at work.

# Discussion of Implications

1. Ethical Implications:
   - For the ethical implications of data collection, storage, and data biases we first need to make sure that customers know that we will collect their data to help train our model that predicts a customer's order. Then for storage, we need to make sure that customers have a way to remove their data if they ask/opt-out. Finally for data biases we need to understand the demographics of our customers might not be representative of the population of the country or even the population at a different college. So when we consider creating a model, we might want to have a seperate model for each location since different colleges might have different biases for what to order and features may be more/less important depending on the population.

2. Buisness Outcome Implications:
   - For the buisness outcome implications we first need to make sure that there is some way to collect data that doesn't require a lot of time or money. The easiest way to collect the data needed to train our model would probably have customers answer a survey after they order in which they give us their information for the features we are considering (Year, Major, etc.). We could then give customers that complete this survey a future discount, that way customers have an incentive and it shouldn't be too expensive for the buisness to implement. Next, we want to make sure that the business is storing the data securly and safely. If the company was storing personal data unsafely they could potentially be sued in the future, and that is obviously not good! The best way to safe is securely would be to de-personalize the data and make sure passwords and other cybersecurity precautions are up to date and taken care of. Also another important thing to consider for storage is the cost, and making sure that a business is storing data in such a way that it won't cost them a fortune. For example, a buisness shouldn't store millions of data points since that type of data storage can be expensive and most likely isn't actually making the model (for predicting customers orders) more accurate/better. Finally, understanding the biases in the data is important for the buisness as being aware of the biases can prevent alienating the current customer base. Buisnesses should understand that machine learning models are a useful tool, but before making any decisions, understand that there might be blindspots in the data and extra caution should be taken.
  
3. Technical Implications:
   - For the technical implications, assuming that we will do data collection in the form of a voluntary survery, we need to make sure that the survey properly sends the results to some CSV file, or a spreadsheet. Of course problems might arise if there are a lot of customers trying to answer/complete the survey, so we should consider what the workload will be. If its light using 3rd party software for the survey and spreadsheet might not be a problem, but if we know that a lot of customers will try to complete the survey at a time, we might consider in house solutions. Then for storage we want to choose an appropriate model that won't take a lot of space, or at least has a reasonable space complexity. This will be useful not only for lowering storage cost, but also for making our model run faster. Finally when considering data biases we have to make sure our model doesn't overfit the given data. What I mean by that is that we can assume that there will be some sort of biases inside the data, and we want our model to try an ignore these biases. Luckily by using certain ML techniques and creating validation sets for our model we can lower the bias in the model.
  
# Model

Here we have the 
