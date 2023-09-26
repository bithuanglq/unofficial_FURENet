# unofficial_FURENet
An unofficial implement of FURENet in Geophysical Research Letters paper: [Nowcasting of convective development by incorporating polarimetric radar variables into a deep‐learning model](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2021GL095302)

Encode each input variable separately, then merge the outputs of the encoder. After passing through the Squeeze and Extraction module, it is then decoded to obtain predictions containing spatial and temporal information.
