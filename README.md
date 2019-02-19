# ml-capstone-submission

Capstone Proposal
Joe Kindle
February 10, 2019

Proposal

Domain Background
I will be using a dataset with employee reviews to predict if a given employee would be satisfied at each of the 6 companies included the dataset (Google, Amazon, Facebook, Netflix, Apple, Microsoft).  I will be doing my prediction mainly based on the reviewer's review text, and will incorporate other factors from the dataset where possible.
As an employee in the tech world, I am very interested to see what people think about working at the largest and most well-regarded tech companies around.  
Problem Statement
Many software developers consider employment at one of the FAANG companies the "Holy Grail" of jobs.  These companies tend to treat their employees extremely well and do a lot of cutting-edge work in the field.  Of course, nothing is perfect, and a position at one of these companies is not for everyone.  I will attempt to predict if a given employee is satisfied with their time at their company via Naïve Bayes language processing run against their pros and cons review text and overall scoring.  I also would like to incorporate Glassdoor's scoring system (detailed and overall scoring, helpful count) in my analysis and see if that can improve my predictions.
One implementation of Naïve Bayes is the Gaussian method, which computes probability as follows:
				
From <https://scikit-learn.org/stable/modules/naive_bayes.html#gaussian-naive-bayes> 




Datasets and Inputs
The dataset I will be using is from Kaggle and was scraped from the Glassdoor review site by Kaggle user Peter Sunga (https://www.kaggle.com/petersunga/google-amazon-facebook-employee-reviews).  Glassdoor is by far the best known employer review site, and is certain to have the best data on this topic.  I am not certain on how the data was scraped from Glassdoor, but my best guess is that the data was pulled off the official Glassdoor API.  All of the data in the set is appropriately anonymized; Glassdoor does not tell you the individual that made the review, just where that person worked, when the review was made, and what the reviewer's title was at the time of review.
The dataset contains the following columns:
	1) index
	2) company: Company Name
	3) location: Location of Company
	4) date: Review Date
	5) job-title
	6) summary
	7) pros: Employee Review (pros)
	8) cons: Employee Review (cons)
	9) advise-to-mgmt: Advise to management
	10) overall-ratings: 1-5
	11) work-balance-starts: Work Life Balance (1-5)
	12) culture-values: Culture and Values (1-5)
	13) career-opportunities-stars: Career Opportunities (1-5)
	14) comp-benefit-stars: Compensation/Benefits (1-5)
	15) senior-management-stars: Senior Management (1-5)
	16) helpful-count: # of Helpful Ratings on the Review
	17) link
I will be focused on the pros, cons, job-title, and overall-rating columns for my Naïve Bayes analysis.  I see some potential for improving my predictions via Glassdoor's rating system columns and maybe the job-title column.
Solution Statement

Using a Naïve Bayes predictor, I will attempt to predict if a given employee was satisfied with their employment.  Then, I will make a confusion matrix to visualize the performance of my algorithm with regard to the baseline.  

Benchmark Model

As a baseline model, I will use random guessing with a 70/30 distribution.  Based on some initial testing on the set, I estimate that roughly 70% of employees at FAANG companies are satisfied with their employment.  I will make a confusion matrix with the results and compare it to my solution results.

As a second baseline model, I will consider any employee with "Current Employee" in their job-title as isSatisfied = 1 and "Former Employee" as isSatisfied = 0.  Inspection of the data indicates that this is a poor predictor of satisfaction (many false negatives).

Evaluation Metrics

I will compare my benchmark model with my Naïve Bayes model via confusion matrix.  The table will outline the performance of the isSatisfied prediction as seen below:
	1) true positive (TP): correctly predicted true
	2) true negative (TN): correctly predicted false
	3) false positive (FP): incorrectly predicted true (type I error)
	4) false negative (FN): incorrectly predicted false (type II error)

	Predicted Satisfied	Predicted Unsatisfied
Satisfied	True Positive	False Negative
Unsatisfied	False Positive	True Negative

Project Design

First, I will pre-process the dataset by:
	1) Omitting all columns except pros, cons, job-title, and overall-rating.  
	2) Filtering out stopwords because these words are generally not helpful for the purposes of natural language processing.  Be careful so that the words removed are truly unhelpful.
	3) Computing an "isSatisfied" score using the reviewer's overall rating.  Initially this will be 1 if overall-rating is greater than 3 and 0 otherwise.  This indicates whether the reviewer is satisfied with their time at the employer, and will be used to train my Naïve Bayes predictor.

Then, I will use sklearn.feature_extraction to convert the text into something my Naïve Bayes predictor can understand.  There are some options here and I plan to start with DictVectorizer (one-hot encoding) and CountVectorizer (Bag of Words).  FeatureHasher (implementation of the "hashing truck") seems to have performance improvements at the cost of some implementation/debugging complexity.

Next, I will create my baseline model via 70/30 random guessing and the associated confusion matrix.

Then, I will run my transformed text data through an sklearn.naive_bayes implementation (potentially) and make a prediction for each employee's text review.

Finally, I will compare the resulting confusion matrices and attempt to make improvements to my algorithm.  Some potential areas of improvement:
	- Incorporating new columns beyond the pros and cons.  For example, the job-title column states whether the employee is a current or former employee.  This is not a 
	- Using a feature extraction or naïve bayes implementation that works better for my problem.  Likely some trial and error here.


