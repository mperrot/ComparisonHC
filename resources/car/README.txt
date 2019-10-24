----------------------------------
60 Cars Data
----------------------------------

AUTHORS:
Matthäus Kleindessner and Ulrike von Luxburg
University of Tübingen

For questions or comments please contact Ulrike von Luxburg: 
luxburg@informatik.uni-tuebingen.de



We collected ordinal data in the form of statements "Object A is the most central object 
within the triple of objects (A,B,C)" for 60 images of different cars in an online survey. 
We refer to the 60 images as "car data set". Using only the collected statements, we can 
solve several machine learning problems on the car data set. See our paper (cited below) 
for details.


The "60 Cars Data" consists of two parts:

1. The car data set consisting of 60 images of different cars. All images were found on 
Wikimedia Commons (https://commons.wikimedia.org) and have been explicitly released into 
the public domain by their authors. In our paper, we divided the car data set into four 
subclasses as follows:

ORDINARY CARS: 2 6 7 8 9 10 11 12 16 17 25 32 35 36 37 38 39 41 44 45 46 55 58 60

SPORTS CARS: 15 19 20 28 40 42 47 48 49 50 51 52 54 56 59

OFF-ROAD/SPORT UTILITY VEHICLES: 1 3 4 5 13 14 18 22 24 26 27 29 31 33 34 43 57

OUTLIERS: 21 23 30 53


2. Ordinal data in the form of statements "Object A is the most central object within the 
triple of objects (A,B,C)". The ordinal data was collected via an online survey in October 
and November 2015. See our paper (cited below) for details.
   
The main file is survey_data.csv. It provides detailed information about the collected 
data. Each row corresponds to one collected statement and consists of six numbers 
a,b,c,d,e,f (separated by commas) with the following meaning:
* a: The number of the round of the survey. Each round was done by one survey 
     participant, but some participants might have contributed several rounds (maybe at 
     various times).
* b: The image that was chosen to be the most central one within the triple of 
     images (c,d,e).
* c,d,e: The triple of images within which b is the most central one. In the survey the 
         three images were shown next to each other on a white screen, c on the left side,
         d in the middle, and e on the right side. 
* f: The response time (in seconds) that it took the participant to choose the most 
     central image. 
   
In the folder "ordinal_data_preprocessed" we provide files extracted from 
survey_data.csv. The files ALL.csv, ALL_REDUCED.csv, T_ALL.csv, and T_ALL_REDUCED.csv 
correspond to the collections ALL, ALL_REDUCED, T_ALL, and T_ALL_REDUCED as described in 
our paper (cited below). Each file consists of rows consisting of three numbers i,j,k 
that encode statements of the kind "Object A is the most central object within the triple
of objects (A,B,C)" as follows: a row i,j,k (we always have j < k) means that image i was 
chosen to be the most central one within the triple of images (i,j,k).



For using (parts of) the data in publications, please cite the following paper:
Matthäus Kleindessner and Ulrike von Luxburg. 
Lens depth function and k-relative neighborhood graph: versatile tools for ordinal data 
analysis.
Journal of Machine Learning Research (JMLR) 18(58):1-52, 2017. 
   	
   	
   	
   	


