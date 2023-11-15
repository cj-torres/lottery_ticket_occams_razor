library(dplyr)

counts = read.csv("F:\\PycharmProjects\\lottery_ticket_occams_razor\\COUNTS.csv")
prelim = read.csv("F:\\PycharmProjects\\lottery_ticket_occams_razor\\PRELIM_DATA_TOY.csv")

joined = inner_join(counts, prelim, by="function.") %>% filter(l0.complexity != Inf)

ggplot(joined, aes(x=l0.complexity, y=freq)) + geom_point() + scale_y_continuous(trans='log10')

ggplot(joined, aes(x=lz.complexity, y=freq)) + geom_point() + scale_y_continuous(trans='log10')

ggplot(joined, aes(x=l0.complexity, y=lz.complexity)) + geom_point()
