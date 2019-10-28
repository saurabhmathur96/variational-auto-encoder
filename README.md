# variational-auto-encoder
Implementation of VAE from Kingma and Welling (2014) and C-VAE from Sohn, Lee and Yan (2015)

## Model

```
x: image
t: embedding vector
x|t = decoder(t); likelihood
t|x = encoder(x); posterior
```
```
t ~ N(0, I)
x|t ~ N(mu(t), sigma(t))
t|x is intractable :( 
```

## Inference
- Variational EM
- REINFORCE Trick
- Local Reparameterization Trick

### REINFORCE Trick
Used to approximate `grad(E[f(x)])`

Key idea: 
```
If y = f(x)
Then grad(y) = y grad(log(y))
```

- makes no assumptions
- variance is high, needs Rao-Blackwellization/Control Variates

### Local Reparameterization Trick

Used to approximate `grad(E[f(x)])`


Key idea
```
f(x), x ~ N(mu, sigma2)

is same as 

f(x), e ~ N(0, 1)
where x = g(e) = mu + sigma2*e 
```

- low variance
- f must be differentiable
- p(x) must be continuous
- there must exist some g for p(x)
