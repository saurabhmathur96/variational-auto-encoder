# variational-auto-encoder
Implementation of Kingma and Welling (2014)

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


