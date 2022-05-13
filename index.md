## Creative Walk Adversarial Networks: Novel Art Generation with Probabilistic Random Walk Deviation from Style Norms

We propose Creative Walk Adversarial Networks
(CWAN) for novel art generation. Quality learning representation of unseen art styles is critical to facilitate the
generation of new artworks. CWAN learns an improved
metric space for generative art by exploring unseen visual spaces with probabilistic random walks. CWAN
constructs a dynamic graph that includes the seen art
style centers and generated samples in the current minibatch. We then initiate a random walk from each art
style center through the generated artworks in the current minibatch. As a deviation signal, we encourage
the random walk to eventually land after T steps in
a feature representation that is difficult to classify as
any of the seen art styles. We investigate the ability
of the proposed loss to generate meaningful novel visual art on the WikiArt dataset. Our experimental results and human evaluations demonstrate that CWAN
can generate novel art that is significantly more preferable compared to strong state-of-the-art methods, including StyleGAN2 and StyleCAN2
