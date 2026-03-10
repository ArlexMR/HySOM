===================================
Using the *General T-Q SOM* in R
===================================

If you prefer working in R and all you need is to classify C–Q loops using the *General T–Q SOM* 
(That is, if you don't need to train a new SOM) you can do it without leaving R. This is how you can do it:

Note that to classify a set of loops using the *General T-Q SOM* you simply need:  

    1. The set of prototypes (loop types in the *General T-Q SOM*)
    2. The distance function (Dynamic Time Warping) to compute the similarity between your loops and each prototype.

Here's how you can get both:

+++++++++++++++++++++++++++++++++++++
1. How to get the prototypes in R
+++++++++++++++++++++++++++++++++++++

The prototypes that represent the loop types of a trained SOM are stored as a multidimensional array. The *General T-Q SOM* 
has dimensions: :math:`8 \times 8 \times 100 \times 2` (has 8 rows and 8 columns for the map and 100 rows and 2 columns for each loop). 
You can download this multidimensional array as a netcdf file from `this link <https://github.com/ArlexMR/HySOM/raw/refs/heads/main/src/hysom/data/GTQSOM_prots.nc>`__. So, first, download the file and save it in your computer. 

Next, make sure you have installed `ncdf4`. If not you can install it with:

.. code:: r

    install.packages("ncdf4")

Then open the file you just downloaded. See below how to do it. Note that in the code below, we use the function `aperm` to re-arrange 
the array so the first two indices refer to the SOM's rows and columns (`map_row`, `map_col`) 
and the next two indices refer to the loop's rows and columns (`loop_row`, `loop_col`). Also, make sure to replace the path in `netcdf_file` 
with the path where you saved the file you downloaded.

.. code-block:: r
   :linenos:
   :emphasize-lines: 3

   library(ncdf4)
   # open netcdf file 
   netcdf_file <- "path/to/your/file/GTQSOM_prots.nc" #replace with your path 
   nc_data <- nc_open(netcdf_file)

   #Get data as array
   varname <- "General_TQSOM_prots"
   data_array <- ncvar_get(nc_data, varname)

   # get current dimensions in data_array
   current_dims <- sapply(nc_data$var[[varname]]$dim, function(d) d$name)

   # permute array to match desired dimensions
   desired_order <- c("map_row", "map_col", "loop_row", "loop_col")
   perm_vector <- match(desired_order, current_dims)
   prototypes <- aperm(data_array, perm_vector)

   nc_close(nc_data)

At this point, you should have the prototypes stored in the variable `prototypes` as a multidimensional array with dimensions: :math:`8 \times 8 \times 100 \times 2`.

++++++++++++++++++++++++++++++++++++++++++
2. How to get the distance function in R  
++++++++++++++++++++++++++++++++++++++++++  

The next thing you need is the distance function to compute the similarity between your loops and each prototype. 
The distance function used to train the *general T-Q SOM* was Dynamic Time Warping (DTW), and I suggest using the same distance function for classification.
You can use whatever DTW implementation you like, just keep in mind that it should be compatible with two-dimensional sequences and that it implements the 
*dependent* warping step pattern as explained in `shokoohi-yekta et al. 2017 <https://doi.org/10.1007/s10618-016-0455-0>`__. Or just use the next implementation 
which is equivalent to the one included in HySOM and was used to train the *General T-Q SOM*.

.. code-block:: r

   sqr_dist <- function(x1, x2) {
      sum((x1 - x2)^2)
   }

   dtw <- function(x, x_prime) {

      # Get lengths of the sequences
      n <- dim(x)[1]
      m <- dim(x_prime)[1]
      
      # Initialize the cost matrix R with zeros
      R <- matrix(0, nrow = n, ncol = m)
      
      for (i in 1:n) {
         for (j in 1:m) {
            R[i, j] <- sqr_dist(x[i, ], x_prime[j, ])
            
            if (i > 1 || j > 1) {
            # Calculate possible predecessors
            prev_i  <- if (i > 1) R[i - 1, j]     else Inf
            prev_j  <- if (j > 1) R[i, j - 1]     else Inf
            prev_ij <- if (i > 1 && j > 1) R[i - 1, j - 1] else Inf
            R[i, j] <- R[i, j] + min(prev_i, prev_j, prev_ij)
            }
         }
      }
   
      # Return the square root of the final accumulated cost
      return(sqrt(R[n, m]))
   }

++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
3. Classify your loops using the prototypes and distance function 
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Now you have both the prototypes and the distance function, so you can classify your loops by computing the distance between each loop and each prototype and assigning each loop to the closest prototype. 
To demonstrate how to do this, I'll use a loop extracted directly from the prototypes and compute the distance from that loop to each prototype.

.. code-block:: r

   compute_dtw_distances <- function(prototypes, loop) {
   distances <- matrix(NA, nrow = 8, ncol = 8)
   for (i in 1:8) {
      for (j in 1:8) {
         distances[i, j] <- dtw(loop, prototypes[i, j , , ])
      }
   }
   return(distances)
   }

   get_BMU <- function(prototypes, loop  ) {
   distances <- compute_dtw_distances(prototypes, loop)
   return(which(distances == min(distances), arr.ind = TRUE))
   }

   get_distance_to_bmu <- function(prototypes, loop) {
   distances <- compute_dtw_distances(prototypes, loop)
   return(min(distances))
   }

   # Let's test it with a loop extracted directly from the prototypes
   loop <- prototypes[2, 3, , ]

   distances <- compute_dtw_distances(prototypes, loop)
   bmu <- get_BMU(prototypes, loop)
   distance_to_bmu <- get_distance_to_bmu(prototypes, loop)

   # Print results
   cat("Distance Matrix:\n")
   print(round(distances, 3))
   cat("\nBMU:\n")
   print(get_BMU(prototypes, loop))
   cat("\nDistance to BMU:\n")
   print(get_distance_to_bmu(prototypes, loop))

The output of the code above should look like this::

   Distance Matrix:
         [,1]  [,2]  [,3]  [,4]  [,5]  [,6]  [,7]  [,8]
   [1,] 1.155 0.653 0.624 0.794 1.075 1.647 2.943 3.905
   [2,] 0.842 0.457 0.000 0.695 0.966 1.405 2.074 3.369
   [3,] 1.132 0.717 0.559 0.728 0.954 1.260 1.954 2.814
   [4,] 1.294 1.208 1.048 1.034 1.103 1.352 1.769 2.471
   [5,] 1.620 1.515 1.380 1.219 1.334 1.380 1.911 2.516
   [6,] 2.117 1.737 1.723 1.616 1.609 1.791 1.940 2.192
   [7,] 3.473 2.671 2.077 1.948 2.167 2.140 2.163 2.608
   [8,] 4.728 3.794 2.693 2.316 2.906 2.731 2.518 2.929

   BMU:
      row col
   [1,]   2   3

   Distance to BMU:
   [1] 0

In this example, the loop we tested was extracted from the prototype located in row 2 and column 3 of the map, 
so it makes sense that the closest prototype (BMU) is located in row 2 and column 3 and that the distance to the BMU is 0.