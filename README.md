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


