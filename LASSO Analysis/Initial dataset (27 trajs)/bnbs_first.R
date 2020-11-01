library(glmnet)

df = read.csv('bnbs.csv', header = TRUE)
data = as.matrix(df)

y = data[,2]
x = data[,3:ncol(data)]
print(dim(x))

## If you do not want intercept, set intercept=F
fit = glmnet(x, y, family = "binomial", intercept = T)
predict(fit, newx = x[1:22,], type = "class", s = c(0.05, 0.01))
plot(fit, xvar = "dev", label = TRUE)
cvfit = cv.glmnet(x, y, family = "binomial", type.measure = "class")
print(cvfit$lambda.min)
print(coef(cvfit, s = "lambda.min"))

tmp_coeffs <- coef(cvfit, s = "lambda.min")
data.frame(name = tmp_coeffs@Dimnames[[1]][tmp_coeffs@i + 1], coefficient = tmp_coeffs@x)
