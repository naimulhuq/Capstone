#Capstone Project: "Airline Twitter Sentiment Analysis" 

#1. Set working directory 

getwd()

working_dir <-  "C:/Users/naimul/Documents"   
working_dir<- "C:/Naimul/Education/Ryerson/CKME 136 Capstone/dataset"

setwd(working_dir)



#2. Import data, check data characteristics and clean  
airline_dataset <- read.csv(file.choose(), header=T, sep=",", check.names=TRUE)

## Check names
names(airline_dataset)

## Check 1st 3 contents of the dataframe 
head(airline_dataset, 3)

## Structure of an object
str(airline_dataset) 

## Class or type of an object
class(airline_dataset) 

## Number of elements or components
length(airline_dataset)

## Remove duplicate data from the dataset
airline1 <- airline_dataset[!duplicated(airline_dataset$text), ]
str(airline1)

write.csv(airline1, 'airline1_clean.csv')



#3. Install required packages
library(sentiment) ## R package with tools for sentiment analysis including Bayesian classifiers for positivity/negativity and emotion classification.
library(tm) ##Text Mining
library(plyr) ## Tools for splitting, applying and combining data
library(Rstem) ## Interface to Snowball implementation of Porter's word stemming algorithm
library(SnowballC) ## An R interface to the C libstemmer library that implements Porter's word stemming algorithm for collapsing words to a common root.
library(ggplot2) ## Data Visualization
library(wordcloud) ## Plot a cloud comparing the frequencies of words across documents. (e.g. comparison cloud) 
library(RColorBrewer) ## Provides color schemes for maps and other graphics.  
library(NLP) ## Basic classes and methods for Natural Language Processing (e.g. tokenize texts). 
library(RWeka) ## Weka is a collection of machine learning algorithms for data mining tasks written in Java, 
               ## Containing tools for data pre-processing, classification, regression, clustering, association rules, and visualization.
library(slam) ## Sparse lightweight arrays and matrices
library(klaR) ## Implementation of naive bayes in R
library(caret) ## Experimental design
library(e1071) ## Confusion matrix
library(MASS) ## Support functions and datasets for Venables and Ripley's MASS
library(lattice) ## Lattice Graphics 



#4. Prepare text for sentiment analysis (Data Cleaning)
air_txt <- airline1$text

## Check first 5 tweets, length and class 
head(air_txt, 5)
length(air_txt)
class(air_txt)

## Remove retweet entities
air_txt = gsub("(RT|via)((?:\\b\\W*@\\w+)+)", "", air_txt) 

## Remove at people
air_txt = gsub("@\\w+", "", air_txt) 

## Remove punctuation
air_txt = gsub("[[:punct:]]", "", air_txt) 

## Remove numbers
air_txt = gsub("[[:digit:]]", "", air_txt) 

## Remove html links
air_txt = gsub("http\\w+", "", air_txt) 

## Remove unnecessary spaces
air_txt = gsub("[ \t]{2,}", "", air_txt)
air_txt = gsub("^\\s+|\\s+$", "", air_txt)

## Define "tolower error handling" function 
try.error = function(x)
{
  # create missing value
  y = NA
  # tryCatch error
  try_error = tryCatch(tolower(x), error=function(e) e)
  # if not an error
  if (!inherits(try_error, "error"))
    y = tolower(x)
  # result
  return(y)
}

## Lower case using try.error with sapply 
air_txt = sapply(air_txt, try.error)

## Remove NAs in air_txt
air_txt = air_txt[!is.na(air_txt)]
names(air_txt) = NULL

## Erasing characters that are not alphabetic, spaces or apostrophes
air_txt <- gsub("[^[:alpha:][:space:]']", " ", air_txt)
air_txt <- gsub("â ", "'", air_txt)
air_txt <- gsub("ã", "'", air_txt)
air_txt <- gsub("ð", "'", air_txt)



#5. Labeled tweets by polarity   

## classify polarity
class_pol = classify_polarity(air_txt, algorithm="voter") ## instead of "bayes" can use "voter" 

## get polarity best fit
polarity = class_pol[,4]

## data frame with results
air_sentiment <- data.frame(text=air_txt, 
                     polarity=polarity, stringsAsFactors=FALSE)

## sort data frame
air_sentiment = within(air_sentiment,
                 polarity <- factor(polarity, levels=names(sort(table(polarity), decreasing=TRUE))))

write.csv(air_sentiment, 'air_sentiment.csv')


## count polarity
table(air_sentiment$polarity)


## plot distribution of polarity

ggplot(air_sentiment, aes(x=polarity)) +
  geom_bar(aes(y=..count.., fill=polarity))+xlab("Polarities") + ylab("Tweet Count") + 
  ggtitle("Sentiment Analysis of Airline Tweets based on Polarity")



#6. Labeled key words by polarity

## separating text by polarity
pol = levels(factor(air_sentiment$polarity))
nemo = length(pol)
pol.docs = rep("", nemo)
for (i in 1:nemo)
{
  tmp = air_txt[polarity == pol[i]]
  pol.docs[i] = paste(tmp, collapse=" ")
}


## remove stopwords
pol.docs <- removeWords(pol.docs, stopwords("english"))

## create corpus
corpus <- Corpus(VectorSource(pol.docs))


## Stem words 
library(SnowballC)
library(tm)

corpus <- tm_map(corpus, stemDocument, language = "english")


## Create Term Document Matrix
tdm <- TermDocumentMatrix(corpus)
colnames(tdm) <- pol

## Convert it to matrix
tdm <- as.matrix(tdm)
class(tdm)

## Save as csv file
write.csv(tdm, 'airline_words_polarity.csv')

## Frequent Words and Association

## Which term has the most occurances
max(apply(tdm,1,sum))  
which(apply(tdm,1,sum)==3276)

## Top 20 terms based on occurances 
which(apply(tdm,1,sum)>600)

## Word Cloud
library(wordcloud)

##  WordCloud
word.freq <- sort(rowSums(tdm), decreasing = T)
wordcloud(words = names(word.freq), freq = word.freq, min.freq = 200,
          random.order = F)
 
## Comparison Cloud
comparison.cloud(tdm, colors = brewer.pal(nemo, "Dark2"),
                 scale = c(3,.5), random.order = FALSE, title.size = 1.5)



#7. Tokenization

## Check type of an object 
class(air_txt)

## Create corpus
corpus1 <- Corpus(VectorSource(air_txt))

## Clean the data
corpus1 <- tm_map(corpus1, tolower)
corpus1 <- tm_map(corpus1, removeNumbers)
corpus1 <- tm_map(corpus1, removePunctuation)
corpus1 <- tm_map(corpus1, removeWords, stopwords("english"))
corpus1 <- tm_map(corpus1, stemDocument, "english")
corpus1 <- tm_map(corpus1, stripWhitespace)
corpus1 <- tm_map(corpus1, PlainTextDocument)

## Create function for Tokenization
UnigramTokenizer <- function(x) NGramTokenizer(x, Weka_control(min = 1, max = 1))

## Tokenize (ngram) 
all.unigrams <- TermDocumentMatrix(corpus1, control = list(tokenize = UnigramTokenizer))

## Create Term Frequency
library(slam)

all.unigrams.freq <- sort(row_sums(all.unigrams), decreasing = TRUE)

## Create the top "101" data frames from the matrices
Top101_terms <- data.frame("Term"=names(head(all.unigrams.freq,101)), "Frequency"=head(all.unigrams.freq,101))
print(Top101_terms)

write.csv(Top101_terms, 'Top101_terms.csv')

## Convert as matrix
Unigrams <- as.matrix(all.unigrams)

class(Unigrams)
nrow(Unigrams)
ncol(Unigrams)

## Save as csv file
write.csv(Unigrams, 'Unigrams.csv')

## Transpose the matrix
Unigrams_transpose<- t(Unigrams)

## Save as csv file
write.csv(Unigrams_transpose, 'Unigrams.csv')

## Convert matrix into dataframe
Unigrams_df<-as.data.frame(Unigrams_transpose)

## Add polarity into unigrams_df 
polarity <-as.character(air_sentiment$polarity)

## Create new data frame with frequently  used 100 unigrams and polarity 
TOP100_Unigrams_df<-data.frame(cbind(                               
                                   flight=Unigrams_df$flight, get=Unigrams_df$get, can=Unigrams_df$can, now=Unigrams_df$now,
                                   just=Unigrams_df$just, help=Unigrams_df$help, service=Unigrams_df$service, thank=Unigrams_df$thank,
                                   thanks=Unigrams_df$thanks, will=Unigrams_df$will, customer=Unigrams_df$customer, time=Unigrams_df$time,
                                   hold=Unigrams_df$hold, amp=Unigrams_df$amp, cant=Unigrams_df$cant, plane=Unigrams_df$plane,
                                   still=Unigrams_df$still, flights=Unigrams_df$flights, need=Unigrams_df$need, one=Unigrams_df$one,
                                   
                                   back=Unigrams_df$back, bag=Unigrams_df$bag, dont=Unigrams_df$dont, call=Unigrams_df$call,
                                   got=Unigrams_df$got, please=Unigrams_df$please, gate=Unigrams_df$gate, delayed=Unigrams_df$delayed,
                                   cancelled=Unigrams_df$cancelled, like=Unigrams_df$like, today=Unigrams_df$today, phone=Unigrams_df$phone,
                                   hour=Unigrams_df$hour, know=Unigrams_df$know, fly=Unigrams_df$fly, airline=Unigrams_df$airline,
                                   guys=Unigrams_df$guys, way=Unigrams_df$way, trying=Unigrams_df$trying, airport=Unigrams_df$airport,
                                   
                                   delay=Unigrams_df$delay, great=Unigrams_df$great, day=Unigrams_df$day, ive=Unigrams_df$ive,
                                   wait=Unigrams_df$wait, going=Unigrams_df$going, waiting=Unigrams_df$waiting, never=Unigrams_df$never,
                                   make=Unigrams_df$make, even=Unigrams_df$even, flying=Unigrams_df$flying, good=Unigrams_df$good,
                                   tomorrow=Unigrams_df$tomorrow, seat=Unigrams_df$seat, change=Unigrams_df$change, last=Unigrams_df$last,
                                   want=Unigrams_df$want, new=Unigrams_df$new, check=Unigrams_df$check, weather=Unigrams_df$weather,
                                   
                                   really=Unigrams_df$really, told=Unigrams_df$told, work=Unigrams_df$work, first=Unigrams_df$first,
                                   take=Unigrams_df$take, another=Unigrams_df$another, travel=Unigrams_df$travel, see=Unigrams_df$see,
                                   agent=Unigrams_df$agent, email=Unigrams_df$email, getting=Unigrams_df$getting, ticket=Unigrams_df$ticket,
                                   bags=Unigrams_df$bags, due=Unigrams_df$due, worst=Unigrams_df$worst, home=Unigrams_df$home,
                                   yes=Unigrams_df$yes, love=Unigrams_df$love, much=Unigrams_df$much, lost=Unigrams_df$lost,
                                   
                                   two=Unigrams_df$two, luggage=Unigrams_df$luggage, baggage=Unigrams_df$baggage, people=Unigrams_df$people,
                                   thats=Unigrams_df$thats, crew=Unigrams_df$crew, united=Unigrams_df$united,  someone=Unigrams_df$someone,                                   
                                   cancel=Unigrams_df$cancel, right=Unigrams_df$right, late=Unigrams_df$late, didnt=Unigrams_df$didnt,  
                                   made=Unigrams_df$made, trip=Unigrams_df$trip, ever=Unigrams_df$ever, number=Unigrams_df$number,    
                                   hours=Unigrams_df$hours, let=Unigrams_df$let, canceled=Unigrams_df$canceled, sitting=Unigrams_df$sitting,
                                   
                                   polarity))             



## Save as csv file                                   
write.csv(TOP100_Unigrams_df, 'TOP100_Unigrams_df.csv')



#8. Naive Bayes Model

## library
library("klaR") ## implementation of naive bayes in R
library("caret") ## experimental design
library("e1071") ## confusion matrix

## Naive Bayes Model
rn_train <- sample(nrow(TOP100_Unigrams_df),
                   floor(nrow(TOP100_Unigrams_df)*0.7))

train <- TOP100_Unigrams_df[rn_train,]
test <- TOP100_Unigrams_df[-rn_train,]
model <- NaiveBayes(polarity~., data=train)

print(model)


## Make predictions
predictions <- predict(model, TOP100_Unigrams_df)

## Summarize results
confusionMatrix(predictions$class, predictions$class)



#9. Data Visualization

## Visualize 50 important terms based on frequency
Top50_terms <- data.frame("Term"=names(head(all.unigrams.freq,50)), "Frequency"=head(all.unigrams.freq,50))

ggplot(Top50_terms, aes(x = Term, y = Frequency)) + geom_bar(stat = "identity") +
  xlab("50 Important Terms") + ylab("Count") + coord_flip()