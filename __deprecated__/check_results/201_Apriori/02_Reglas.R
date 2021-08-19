install.packages(c("tidyverse","dplyr","apriori","arules"))

library(tidyverse)
library(arules)

binary_data <- read_csv('./output/binary_matrix.csv')
binary_data

transaction_data <- as(as.matrix(select(binary_data, -file)), "transactions")
summary(transaction_data)

itemFrequencyPlot(transaction_data, support = 0.0001, cex.names = 0.8)

rules <- apriori(transaction_data, parameter = list(support = 0.005, confidence = 0.5))
rules

summary(rules)
top_20_itemsets <- sort(rules, by = "lift", decreasing = TRUE)
inspect(top_20_itemsets)
