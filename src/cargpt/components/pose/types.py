from jaxtyping import Float
from torch import Tensor

# 6D: x (right), y (down), z (fwd), theta_x, theta_y, theta_z
Pose = Float[Tensor, "... 6"]
