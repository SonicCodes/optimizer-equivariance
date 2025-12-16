## Why is Adam ? :0 

I have sorta always thought the reason Adam outperforms SGD is the adaptive per parameter learning rates, I think this is a good enough explanation for most intents and purposes.

I'd have called it a day but I found this paper which has an interesting perspective over adam's property, it uses the convergence bound of these optimizers and empirical tests on models to make a case to why we would need to use adam over sgd, 

### Smoothness constant ... Convergence Bounds:
- **SGD** under the L2 norm has smoothness constant: $\|\nabla^2 L\|_2$
- **Adam** under the L-inf norm has a smoothness constant: $\|\nabla^2 L\|_{1,1}$

They justified adam's smoothness under the linf norm and derived the convergence bound and smoothness constant from those assumptions, under these assumptions the smoothness constant depends on the sum of entries on the hessian, thus the second order interaction of the gradients of the parameters, the bound ends up including the sum of the per-coordinate curvature of the loss landscape,

SGD's convergence bound on the other hand only depends on the maximum eigenvalue of the hessian, i.e  the maximum curvature of the loss landscape

### Rotating Loss Landscape

Rotating Loss Landscape refers to computing the gradients using rotated parameters (projected using rotation matrix) and applying it on the unrotated parameter, since sgd is rotation equivariant thus the trajectory is consistent, whereas adam is not and struggles to converge.

Intuitively rotating the loss landscape doesn't change the spectral norm, since rotation doesn't change the eigenvalue only the eigenvectors, For adam we operate on per-cordinate basis , thus the rotation smears up the offdiagonals which inturn increase the (1,1)-norm of hessian (with huge caveat i will describe towards the end)

### Neural Networks and Transformers

<img width="973" height="195" alt="Screenshot 2025-12-15 at 6 39 40 AM" src="https://github.com/user-attachments/assets/a0dfb32d-90a0-41d0-8a9b-cede68a58194" />
<img width="953" height="257" alt="Screenshot 2025-12-15 at 6 40 09 AM" src="https://github.com/user-attachments/assets/48c462c8-1385-4905-b796-890d57e50f72" />

The hessian tells us how the loss landscape is affected by moving in any direction on it, almost like curvature/evolution, thus a diagonal hessian would be like, "landscape with 3 dimensions, moving in dimension 1 should not affect dimension 2 or 3".  Its impractical for neural networks to have a diagonal hessian as it would mean each parameter has its own loss component, but ...

They observed that (1,1)-norm of the hessian is often smaller than the d×spectral norm suggesting that for a lot of neural networks the parameters tends to be independent, this also intuitively means NNs seems to like the L∞ geometry, though this is slightly inaccurate, they never directly measured if the off diagonals are smaller than the diagonals of the hessian, but rotation does justifiably smear off-diagonals and the increasing of the (1,1)-norm suggests something close is happening

All in all neural network naturally tend to prefer separable loss landscape evolution, sgd's update forces them to operate on the non-optimal l2 geometry, which forces parameters to have second order interactions which turns out to be suboptimal, Adam's updates treats the loss landscape on the L∞ geometry which is more suitable.


== See `run.py` to experiment with this
