## Script to test the calculation of pipe metrics
## October 2023


# Import packages
import networkx as nx
import pandas as pd
import math
import wntr
import numpy as np
import geopandas as gpd
from shapely import geometry
import libpysal as lps
from esda.moran import Moran, Moran_Local
from splot.esda import lisa_cluster
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.optimize import minimize 

#%% Calculate water distribution network topographic metrics

class WaterNetworkAnalyzer:
    def __init__(self, inp_file_path, path_pipes, weight_bc, weight_cc, weight_bridges):
        self.inp_file_path = inp_file_path
        self.path_pipes = path_pipes
        self.weight_bc = weight_bc
        self.weight_cc = weight_cc
        self.weight_bridges = weight_bridges
        self.wn = wntr.network.WaterNetworkModel(inp_file_path)
        self.G = self.wn.to_graph()
        self.uG = self.G.to_undirected()
        self.sG = nx.Graph(self.uG)
        self.pipe_gdf = None  # Initialize the pipe_gdf as None
        self.bridges = None # Initialize bridges as None
   
    def min_max_scaling(self, series):
        # Convert the series to numeric, setting non-numeric values to NaN
        numeric_series = pd.to_numeric(series, errors='coerce')
        
        # Calculate min and max values from the numeric series
        min_val = numeric_series.min()
        max_val = numeric_series.max()

        # Perform Min-Max scaling on the numeric series
        scaled_series = (numeric_series - min_val) / (max_val - min_val)

        return scaled_series

    def calculate_combined_metric(self, df):
        # Min-Max scaling for BC Av Pipe and CC Av Pipe columns
        df['BC Av Pipe'] = self.min_max_scaling(df['BC Av Pipe'])
        df['CC Av Pipe'] = self.min_max_scaling(df['CC Av Pipe'])

        # Calculate the combined metric using the specified weights
        combined_metric = (
            df['BC Av Pipe'] * self.weight_bc +
            df['CC Av Pipe'] * self.weight_cc +
            df['Bridges'] * self.weight_bridges
        )
        return combined_metric
        
    def analyze_network(self):
        # Calculate articulation points and bridges
        articulation_points = list(nx.articulation_points(self.uG))
        bridges = wntr.metrics.bridges(self.G)
        self.bridges = bridges

        # Mark and plot red pipes that are bridges
        ax = wntr.graphics.plot_network(self.wn, link_attribute=bridges, node_size=0.5)

        # Calculate eccentricity and diameter
        diameter = nx.diameter(self.uG)
        eccentricity = nx.eccentricity(self.uG)

        # Calculate Betweenness centrality and central point dominance
        betweenness_centrality = nx.betweenness_centrality(self.sG)
        central_point_dominance = wntr.metrics.central_point_dominance(self.G)

        # Calculate Closeness centrality
        closeness_centrality = nx.closeness_centrality(self.G)

        # Get pipe list
        pipes = self.wn.pipe_name_list

        # Get upstream and downstream node of each pipe
        node_US = []
        node_DS = []
        for pipe in pipes:
            link = self.wn.get_link(pipe)
            node_US.append(link.start_node_name)
            node_DS.append(link.end_node_name)

        # Create an array containing the upstream and downstream node names for each pipe
        pipe_nodes = np.array(list(zip(pipes, node_US, node_DS)))

        # Get the array length
        nr_pipes = len(pipe_nodes)

        # Add new columns containing the betweenness centrality metrics of each node and the average value for each pipe
        new1 = np.zeros(nr_pipes)
        new2 = np.zeros(nr_pipes)
        new3 = np.zeros(nr_pipes)
        new4 = np.zeros(nr_pipes)
        new5 = np.zeros(nr_pipes)
        new6 = np.zeros(nr_pipes)

        pipe_nodes = np.c_[pipe_nodes, new1, new2, new3, new4, new5, new6]

        for i in range(nr_pipes):
            pipe_nodes[i, 3] = betweenness_centrality[pipe_nodes[i, 1]]
            pipe_nodes[i, 4] = betweenness_centrality[pipe_nodes[i, 2]]
            pipe_nodes[i, 5] = (betweenness_centrality[pipe_nodes[i, 1]] + betweenness_centrality[pipe_nodes[i, 2]]) / 2
            pipe_nodes[i, 6] = closeness_centrality[pipe_nodes[i, 1]]
            pipe_nodes[i, 7] = closeness_centrality[pipe_nodes[i, 2]]
            pipe_nodes[i, 8] = (closeness_centrality[pipe_nodes[i, 1]] + closeness_centrality[pipe_nodes[i, 2]]) / 2
        
        pipe_nodes_df = pd.DataFrame(pipe_nodes, columns=["Pipe", "Upstream", "Downstream", "BC US", "BC DS", "BC Av Pipe", "CC US", "CC DS", "CC Av Pipe"])
        pipe_gdf = gpd.read_file(path_pipes).set_crs('EPSG:2100')

        # Merge pipe_gdf and pipe_nodes_df based on the LABEL and Pipe columns
        pipe_gdf = pipe_gdf.merge(pipe_nodes_df[['Pipe', 'BC Av Pipe', 'CC Av Pipe']], left_on='LABEL', right_on='Pipe')
        pipe_gdf['Bridges'] = 0
        pipe_gdf['Bridges'] = pipe_gdf['LABEL'].isin(self.bridges).astype(int)
        
        # Calculate the combined metric
        pipe_gdf['Combined Metric'] = self.calculate_combined_metric(pipe_gdf)
        
        self.pipe_gdf = pipe_gdf  # Store the resulting pipe_gdf in the instance variable

        return pipe_gdf, bridges
    
    def save_pipe_gdf_shapefile(self, output_path):
        if self.pipe_gdf is not None:
            self.pipe_gdf.to_file(output_path, driver="ESRI Shapefile")
            print(f"Pipe GeoDataFrame saved to {output_path}")
        else:
            print("No data to save. Run analyze_network() first to generate data.")
        

if __name__ == "__main__":
    inp_file_path = r"C:\Users\Panos\Dropbox\EYDAP_Asset Management\Calcs\WP2\inp_files\01_07_2021-GP025-F.inp"
    path_pipes = r"C:\Users\Panos\Dropbox\EYDAP_Asset Management\Calcs\WP2\shapefiles\Pipes_WG_export.shp"
    analyzer = WaterNetworkAnalyzer(inp_file_path, path_pipes, weight_bc = 0.33, weight_cc = 0.33, weight_bridges = 0.33)
    analyzer.analyze_network()
    pipe_gdf, bridges = analyzer.analyze_network()

    analyzer.save_pipe_gdf_shapefile(r"C:\Users\Panos\Dropbox\EYDAP_Asset Management\Calcs\WP2\shapefiles\Pipes_WG_export_with_metrics.shp")

#%% Perform Spatial Autocorrelation Analysis (OOP)

class SpatialAutocorrelationAnalysis:
    def __init__(self, pipe_shapefile_path, failures_shapefile_path, weight_avg_combined_metric, weight_failures): 
        self.pipe_shapefile_path = pipe_shapefile_path
        self.failures_shapefile_path = failures_shapefile_path
        self.pipe_gdf = None
        self.failures_gdf = None
        self.fishnet_failures = None
        # Weights for each column
        self.weight_avg_combined_metric = weight_avg_combined_metric
        self.weight_failures = weight_failures

    def read_shapefiles(self):
        self.pipe_gdf = gpd.read_file(self.pipe_shapefile_path).set_crs('EPSG:2100')
        self.failures_gdf = gpd.read_file(self.failures_shapefile_path).set_crs('EPSG:2100')

    def create_fishnet(self, square_size):
        total_bounds = self.pipe_gdf.total_bounds
        minX, minY, maxX, maxY = total_bounds
        x, y = (minX, minY)
        geom_array = []
        
        while y <= maxY:
            while x <= maxX:
                geom = geometry.Polygon([(x, y), (x, y + square_size), (x + square_size, y + square_size), (x + square_size, y), (x, y)])
                geom_array.append(geom)
                x += square_size
            x = minX
            y += square_size
        
        fishnet = gpd.GeoDataFrame(geom_array, columns=['geometry']).set_crs('EPSG:2100')
        return fishnet

    def spatial_autocorrelation_analysis(self):
        results = []

        for square_size in range(100, 1100, 100):
            fishnet = self.create_fishnet(square_size)

            # Perform spatial join to count failures per feature of the fishnet
            fishnet_failures = fishnet.join(
                gpd.sjoin(self.failures_gdf, fishnet).groupby("index_right").size().rename("failures"),
                how="left",
            )
            
            fishnet_failures = fishnet_failures.dropna()

            # Perform spatial join with pipe_gdf to calculate the average Combined Metric per fishnet square
            pipe_metrics = gpd.sjoin(self.pipe_gdf, fishnet_failures, how='inner', predicate='intersects')
            avg_metrics_per_square = pipe_metrics.groupby("index_right")['Combined M'].mean()

            # Add the average Combined Metric to the fishnet_failures GeoDataFrame
            fishnet_failures['avg_combined_metric'] = fishnet_failures.index.map(avg_metrics_per_square)
              
            # Standardize the 'failures' column from 0 to 1
            min_failures = fishnet_failures['failures'].min()
            max_failures = fishnet_failures['failures'].max()
            fishnet_failures['failures_standardized'] = (fishnet_failures['failures'] - min_failures) / (max_failures - min_failures)

            # Add the weighted average column
            fishnet_failures['weighted_avg'] = (
                fishnet_failures['avg_combined_metric'] * self.weight_avg_combined_metric +
                fishnet_failures['failures_standardized'] * self.weight_failures
            ) / (self.weight_avg_combined_metric + self.weight_failures)

            # Store fishnet_failures as an instance variable
            self.fishnet_failures = fishnet_failures

            # Create static choropleth maps (Equal intervals, Quantiles, Natural Breaks)
            # Ensure that the create_choropleth_maps method can handle the new column
            self.create_choropleth_maps(fishnet_failures, square_size)

            # Calculate global Moran's I and store results
            y = fishnet_failures['weighted_avg']
            w = lps.weights.Queen.from_dataframe(fishnet_failures, use_index = False)
            w.transform = 'r'
            moran = Moran(y, w)
            results.append((square_size, moran.I, moran.p_sim, moran.z_sim))

        # Print results
        self.print_results(results)
        return results

    def create_choropleth_maps(self, fishnet_failures, square_size):
        fig, ax = plt.subplots(figsize=(12, 10))
        fishnet_failures.plot(column='weighted_avg', scheme='equal_interval', k=10, cmap='RdYlGn_r', legend=True, ax=ax,
                          legend_kwds={'loc':'center left', 'bbox_to_anchor':(1,0.5), 'fmt':"{:.2f}", 'interval':True})
        fishnet_failures.boundary.plot(ax=ax)
        plt.title(f'Average criticality metric per fishnet cell (size = {square_size} m x {square_size} m), Equal intervals', fontsize = 18)
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
        # Create a static choropleth map of the failure number per grid cell of the fishnet (Quantiles)
        fig, ax = plt.subplots(figsize=(12, 10))
        fishnet_failures.plot(column='weighted_avg', scheme='quantiles', k=10, cmap='RdYlGn_r', legend=True, ax=ax,
                          legend_kwds={'loc':'center left', 'bbox_to_anchor':(1,0.5), 'fmt':"{:.2f}", 'interval':True})
        fishnet_failures.boundary.plot(ax=ax)
        plt.title(f'Average criticality metric per fishnet cell (size = {square_size} m x {square_size} m), Quantiles', fontsize = 18)
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
        # Create a static choropleth map of the failure number per grid cell of the fishnet (Natural Breaks)
        fig, ax = plt.subplots(figsize=(12, 10))
        fishnet_failures.plot(column='weighted_avg', scheme='natural_breaks', k=10, cmap='RdYlGn_r', legend=True, ax=ax,
                          legend_kwds={'loc':'center left', 'bbox_to_anchor':(1,0.5), 'fmt':"{:.2f}", 'interval':True})
        fishnet_failures.boundary.plot(ax=ax)
        plt.title(f'Average criticality metric per fishnet cell (size = {square_size} m x {square_size} m), Natural Breaks', fontsize = 18)
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    def print_results(self, results):
        # Print results here as in your original code
        for size, moran_i, p_value, z_score in results:
            print(f"Square Size: {size} m")
            print(f"Moran's I value: {moran_i} m")
            print(f"Moran's I p-value: {p_value}")
            print(f"Moran's I z-score: {z_score}")
            print()
            
        # Extract data for plotting
        square_sizes, moran_values, p_values, z_scores = zip(*results)

        # Create a plot with multiple y-axes
        fig, ax1 = plt.subplots(figsize=(10, 6))

        # Plot Moran's I on the left y-axis
        ax1.plot(square_sizes, moran_values, 'b-', label="Moran's I", marker='o')
        ax1.set_xlabel("Square Size (m)", fontsize=12)
        ax1.set_ylabel("Moran's I", color='b', fontsize=12)
        ax1.tick_params(axis='y', labelcolor='b')

        # Create right y-axes for p-value and z-score
        ax2 = ax1.twinx()
        ax2.plot(square_sizes, p_values, 'r-', label="p-value", marker='s')
        ax2.set_ylabel("p-value", color='r', fontsize=12)
        ax2.tick_params(axis='y', labelcolor='r')

        ax3 = ax1.twinx()
        ax3.spines['right'].set_position(('outward', 60))
        ax3.plot(square_sizes, z_scores, 'g-', label="z-score", marker='^')
        ax3.set_ylabel("z-score", color='g', fontsize=12)
        ax3.tick_params(axis='y', labelcolor='g')

        # Add grid lines with different colors
        ax1.grid(True, alpha=0.7, color='b')  # Moran's I grid in blue
        ax2.grid(True, linestyle='--', alpha=0.7, color='r')  # p-value grid in red
        ax3.grid(True, linestyle='--', alpha=0.7, color='g')  # z-score grid in green

        # Add labels for each y-axis
        ax1.set_title("Moran's I, p-value, and z-score vs. Square Size", fontsize=16)
        ax1.set_xlabel("Square Size (m)", fontsize=12)

        # Show the legend
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        lines3, labels3 = ax3.get_legend_handles_labels()
        ax1.legend(lines + lines2 + lines3, labels + labels2 + labels3, loc='upper left')

        plt.tight_layout()
        plt.show()

    def find_best_square_size(self, results):
        # Find the square size where Moran's I is maximized and p-value is less than 0.05
        best_square_size = None
        max_moran_i = None
        
        for size, moran_i, p_value, z_score in results:
            if p_value < 0.05:
                if max_moran_i is None or moran_i > max_moran_i:
                    max_moran_i = moran_i
                    best_square_size = size
        
        print(f"Best Square Size: {best_square_size} m")
        print(f"Max Moran's I: {max_moran_i}")  
        return best_square_size

    def optimal_fishnet(self, best_square_size):
        # Perform the analysis and store results for the optimal square size
        total_bounds = self.pipe_gdf.total_bounds
        minX, minY, maxX, maxY = total_bounds
        x, y = (minX, minY)
        geom_array = []
        # Create a fishnet
        x, y = (minX, minY)
        geom_array = []
         
        # Polygon Size
        square_size = best_square_size
        while y <= maxY:
            while x <= maxX:
                geom = geometry.Polygon([(x,y), (x, y+square_size), (x+square_size, y+square_size), (x+square_size, y), (x, y)])
                geom_array.append(geom)
                x += square_size
            x = minX
            y += square_size
         
        fishnet = gpd.GeoDataFrame(geom_array, columns=['geometry']).set_crs('EPSG:2100')
        fishnet.to_file(r"C:\Users\Panos\Dropbox\EYDAP_Asset Management\Calcs\WP2\shapefiles\fishnet_grid_opt.shp")
        
        # Perform spatial join to count failures per feature of the fishnet
        self.fishnet_failures = fishnet.join(
            gpd.sjoin(self.failures_gdf, fishnet, how='inner', predicate='intersects').groupby("index_right").size().rename("failures"),
            how="left",
        )
        
        self.fishnet_failures = self.fishnet_failures.dropna()

        # Perform spatial join with pipe_gdf to calculate the average Combined Metric per fishnet square
        pipe_metrics = gpd.sjoin(self.pipe_gdf, self.fishnet_failures, how='inner', predicate='intersects')
        avg_metrics_per_square = pipe_metrics.groupby("index_right")['Combined M'].mean()

        # Add the average Combined Metric to the fishnet_failures GeoDataFrame
        self.fishnet_failures['avg_combined_metric'] = self.fishnet_failures.index.map(avg_metrics_per_square)
          
        # Standardize the 'failures' column from 0 to 1
        min_failures = self.fishnet_failures['failures'].min()
        max_failures = self.fishnet_failures['failures'].max()
        self.fishnet_failures['failures_standardized'] = (self.fishnet_failures['failures'] - min_failures) / (max_failures - min_failures)

        # Add the weighted average column
        self.fishnet_failures['weighted_avg'] = (
            self.fishnet_failures['avg_combined_metric'] * self.weight_avg_combined_metric +
            self.fishnet_failures['failures_standardized'] * self.weight_failures
        ) / (self.weight_avg_combined_metric + self.weight_failures)
        
        self.fishnet_failures = self.fishnet_failures.dropna()
        
        self.fishnet_failures.to_file(r"C:\Users\Panos\Dropbox\EYDAP_Asset Management\Calcs\WP2\shapefiles\fishnet_grid_opt_failures.shp")
        
        ## Measures of spatial autocorrelation: spatial similarity and attribute similarity

        # Spatial similarity, measured by spatial weights, shows the relative strength of a relationship between pairs of locations

        # Here we compute spatial weights using the Queen contiguity (8 directions)
        w = lps.weights.Queen.from_dataframe(self.fishnet_failures, use_index = False)
        w.transform = 'r'

        # Attribute similarity, measured by spatial lags, is a summary of the similarity (or dissimilarity) of observations for a variable at different locations
        # The spatial lag takes the average value in each weighted neighborhood
        self.fishnet_failures['weighted_fail'] = lps.weights.lag_spatial(w, self.fishnet_failures['weighted_avg'])

        # Global spatial autocorrelation with Moran’s I statistics
        # Moran’s I is a way to measure spatial autocorrelation. 
        # In simple terms, it’s a way to quantify how closely values are clustered together in a 2-D space

        # Moran’s I Test uses the following null and alternative hypotheses:
        # Null Hypothesis: The data is randomly dispersed.
        # Alternative Hypothesis: The data is not randomly dispersed, i.e., it is either clustered or dispersed in noticeable patterns.

        # The value of Moran’s I can range from -1 to 1 where:

        # -1: The variable of interest is perfectly dispersed
        # 0: The variable of interest is randomly dispersed
        # 1: The variable of interest is perfectly clustered together
        # The corresponding p-value can be used to determine whether the data is randomly dispersed or not.

        # If the p-value is less than a certain significance level (i.e., α = 0.05), 
        # then we can reject the null hypothesis and conclude that the data is 
        # spatially clustered together in such a way that it is unlikely to have occurred by chance alone.

        y = self.fishnet_failures['weighted_avg']
        moran = Moran(y, w)
        print(f"Moran's I value: {moran.I}\np-value: {moran.p_sim}\nZ-score: {moran.z_sim}")
        

    def local_spatial_autocorrelation(self):
        
        if self.fishnet_failures is None:
            raise ValueError("You need to run spatial_autocorrelation_analysis first to populate fishnet_failures.")
        
        
        # Perform the local spatial autocorrelation analysis
        y = self.fishnet_failures['weighted_avg']
        w = lps.weights.Queen.from_dataframe(self.fishnet_failures, use_index = False)
        w.transform = 'r'
        # Local spatial autocorrelation with Local Indicators of Spatial Association (LISA) statistics
        # While the global spatial autocorrelation can prove the existence of clusters, 
        # or a positive spatial autocorrelation between the listing price and their neighborhoods, 
        # it does not show where the clusters are. 
        # That is when the local spatial autocorrelation resulted from Local Indicators of Spatial 
        # Association (LISA) statistics comes into play.

        ## Link of main idea: https://github.com/ThucDao/ExploratorySpatialDataAnalysis/blob/main/Exploratory%20Spatial%20Data%20Analysis.ipynb
        ## REMOVE THE SOURCE IN LATER VERSIONS

        # In general, local Moran’s I values are interpreted as follows:

        # Negative: nearby cases are dissimilar or dispersed e.g. High-Low or Low-High
        # Neutral: nearby cases have no particular relationship or random, absence of pattern
        # Positive: nearby cases are similar or clustered e.g. High-High or Low-Low
        # The LISA uses local Moran's I values to identify the clusters in localized map regions and categorize the clusters into five types:

        # High-High (HH): the area having high values of the variable is surrounded by neighbors that also have high values
        # Low-Low (LL): the area having low values of the variable is surrounded by neighbors that also have low values
        # Low-High (LH): the area having low values of the variable is surrounded by neighbors that have high values
        # High-Low (HL): the area having high values of the variable is surrounded by neighbors that have low values
        # Not Significant (NS)

        # High-High and Low-Low represent positive spatial autocorrelation, while High-Low and Low-High represent negative spatial correlation.

        # Create a LISA cluster map
        moran_local = Moran_Local(y, w)

        fig, ax = plt.subplots(figsize=(12,10))
        lisa_cluster(moran_local, self.fishnet_failures, p=1,  ax=ax, legend=True,
                     legend_kwds={'loc':'center left', 'bbox_to_anchor':(1,0.5), 'fmt':"{:.0f}"})
        self.fishnet_failures.boundary.plot(ax=ax)
        plt.title('LISA Cluster Map for average criticality metric per fishnet cell', fontsize = 18)
        plt.tight_layout()
        plt.show()

        # Create a data frame containing the number of failures and the local moran statistics
        fishnet_grid_stats_fails = pd.DataFrame(self.fishnet_failures) 
        fishnet_grid_stats_fails["Local Moran's I (LISA)"] = moran_local._statistic

        # Define a function to get the LISA cluster label
        def get_lisa_cluster_label(val):
            if val == 1:
                return "HH"
            elif val == 2:
                return "LH"
            elif val == 3:
                return "LL"
            elif val == 4:
                return "HL"
            else:
                return "NS"

        # Calculate LISA cluster labels for significant clusters
        cluster_labels = [get_lisa_cluster_label(val) for val in moran_local.q]

        # Add a new column with cluster labels for significant clusters (or "NS" for non-significant clusters)
        fishnet_grid_stats_fails['Cluster_Label'] = cluster_labels

        # Sort the cells according to the cluster label and weighted metric
        sorted_fishnet_df = fishnet_grid_stats_fails.sort_values(by=['Cluster_Label', 'weighted_avg'], ascending=[True, False])

        # Spatially join the fishnet grid cells and pipes
        # First, add an explicit 'fishnet_index' column to the fishnet_failures GeoDataFrame
        self.fishnet_failures['fishnet_index'] = self.fishnet_failures.index
        
        # Now perform the spatial join using this new 'fishnet_index' column
        spatial_join = gpd.sjoin(self.fishnet_failures, self.pipe_gdf, predicate='intersects')
        
        # Group the results by 'fishnet_index'
        grouped = spatial_join.groupby('fishnet_index')
        
        # Create a dictionary to store the results
        results_pipe_clusters = {}
        
        # Iterate through each group (fishnet cell) and collect the associated pipe labels
        for fishnet_index, group_data in grouped:
            pipe_labels = group_data['LABEL'].tolist()
            results_pipe_clusters[fishnet_index] = pipe_labels
        
        # Now, the 'results_pipe_clusters' dictionary contains the pipe labels for each fishnet cell with the fishnet index as the key
        
        return sorted_fishnet_df, results_pipe_clusters 

    def run_analysis(self):
        self.read_shapefiles()
        results = self.spatial_autocorrelation_analysis()
        best_square_size = self.find_best_square_size(results)
        self.optimal_fishnet(best_square_size)
        self.local_spatial_autocorrelation()

# Usage
if __name__ == '__main__':
    pipe_shapefile_path = r"C:\Users\Panos\Dropbox\EYDAP_Asset Management\Calcs\WP2\shapefiles\Pipes_WG_export_with_metrics.shp"
    failures_shapefile_path = r"C:\Users\Panos\Dropbox\EYDAP_Asset Management\Calcs\WP2\shapefiles\Vlaves_Combined.shp"
    analysis = SpatialAutocorrelationAnalysis(pipe_shapefile_path, failures_shapefile_path, weight_avg_combined_metric = 0.5, weight_failures = 0.5)
    analysis.run_analysis()
    
# Run the analysis and capture the return values
results = analysis.local_spatial_autocorrelation()

# Extract the two return values
sorted_fishnet_df, results_pipe_clusters = results

# Save the results in shapefile
sorted_fishnet_gdf = gpd.GeoDataFrame(sorted_fishnet_df).set_crs('EPSG:2100')
sorted_fishnet_gdf.to_file(r"C:\Users\Panos\Dropbox\EYDAP_Asset Management\Calcs\WP2\shapefiles\fishnet_sorted.shp")


#%% Perform Pipe replacement scheduling

# Read the pipe shapefile
path_pipes = r"C:\Users\Panos\Dropbox\EYDAP_Asset Management\Calcs\WP2\shapefiles\Pipes_WG_export_with_metrics.shp"
pipes_gdf = gpd.read_file(path_pipes).set_crs('EPSG:2100')

# Read the sorted fishnet shapefile
path_fishnet = r"C:\Users\Panos\Dropbox\EYDAP_Asset Management\Calcs\WP2\shapefiles\fishnet_sorted.shp"
fishnet_gdf = gpd.read_file(path_fishnet).set_crs('EPSG:2100')

#Keep the n row of the fishnet gdf
# Define the row number you want to keep (e.g., row 1)
row_number_to_keep = 1

# Extract the specific row based on the row number
fishnet_row = fishnet_gdf.iloc[row_number_to_keep-1]

# Get the index of the specific cell
cell_index = fishnet_row['fishnet_in']

# Get the list of pipes contained in the sepcific cell
pipes_cell = results_pipe_clusters[cell_index]

# Create a data frame containing only the pipes of the specific cell
pipes_gdf_cell = pipes_gdf[pipes_gdf['LABEL'].isin(pipes_cell)]
# Reset the index of the data frame
pipes_gdf_cell = pipes_gdf_cell.reset_index()
pipes_gdf_cell = pipes_gdf_cell.drop(columns=['index'])

# Create a dictionary of the pipe age per material
pipe_materials = {
    'Asbestos Cement': 50,
    'Steel': 40,
    'PVC': 20,
    'HDPE': 15
}

# Fill a new column with the pipe age
pipes_gdf_cell['Pipe Age'] = pipes_gdf_cell['MATERIAL'].map(pipe_materials)
pipes_gdf_cell = pipes_gdf_cell[['D', 'LABEL', 'MATERIAL', 'USER_L','Pipe Age', 'ID']]

# Create function to find the ideal optimal replacement age of a single pipe 
def single_opt_age(pipe_id, pipe_age, pipe_diam, CP, Cr, time_span):
    # pipe_id: the id of the pipe
    # pipe_age: the pipe age at the beginning of the planning period (years)
    # pipe_diam: the pipe diameter (mm)
    # CP:the pipe replacement cost (€/km)
    # Cr: the pipe repair cost (€/failure)
    
    #create empty data frame containing the necessary columns
    p_df = pd.DataFrame(index=range(1, time_span + 1), 
                                    columns=['CI', 'Age', 'Fr', 
                                             'SFr/t', 'CR', 'LCC'])
    #fill the columns with the respective values
    for i in range(1, time_span +1):
        p_df.loc[i, 'Age'] = pipe_age + i
        p_df.loc[i, 'CI'] = CP/p_df.loc[i, 'Age']
        p_df.loc[i, 'Fr'] = 0.109 * math.e ** \
                            (-0.0064*pipe_diam) * \
                            p_df.loc[i, 'Age'] ** 1.377
        p_df.loc[i, 'SFr/t'] = p_df['Fr'].loc[:i].sum()/i
        p_df.loc[i, 'CR'] = p_df.loc[i, 'SFr/t'] * Cr
        p_df.loc[i, 'LCC'] = p_df.loc[i, 'CI'] + p_df.loc[i, 'CR']  
    t_star = pd.to_numeric(p_df['LCC'], downcast='float').idxmin()
    #return optimal replacement time for single pipe 
    return t_star, p_df['LCC'].min()

# Create function to find the ideal LCC of a single pipe at time t 
def single_opt_lcc(pipe_id, pipe_age, pipe_diam, CP, Cr,time_span, t):
    # pipe_id: the id of the pipe
    # pipe_age: the pipe age at the beginning of the planning period (years)
    # pipe_diam: the pipe diameter (mm)
    # CP:the pipe replacement cost (€/km)
    # Cr: the pipe repair cost (€/failure)
    
    #create empty data frame containing the necessary columns
    p_df = pd.DataFrame(index=range(1, time_span + 1), 
                                    columns=['CI', 'Age', 'Fr', 
                                             'SFr/t', 'CR', 'LCC'])
    #fill the columns with the respective values
    for i in range(1, time_span +1):
        p_df.loc[i, 'Age'] = pipe_age + i
        p_df.loc[i, 'CI'] = CP/p_df.loc[i, 'Age']
        p_df.loc[i, 'Fr'] = 0.109 * math.e ** \
                            (-0.0064*pipe_diam) * \
                            p_df.loc[i, 'Age'] ** 1.377
        p_df.loc[i, 'SFr/t'] = p_df['Fr'].loc[:i].sum()/i
        p_df.loc[i, 'CR'] = p_df.loc[i, 'SFr/t'] * Cr
        p_df.loc[i, 'LCC'] = p_df.loc[i, 'CI'] + p_df.loc[i, 'CR']  
        
    return p_df.loc[t, 'LCC']

# Create function to find the ideal LCC of the whole network when an array 
# containing the single pipe replacement time is given:
def lcc_tot_net(repl_time_array, pipe_table,time_span):
    t_rep_count = 0
    lcc_tot = 0 
    for p_id in pipe_table['ID']:
        t_rep = repl_time_array[t_rep_count]
        p_age = pipe_table.loc[pipe_table['ID'] == p_id, 'Pipe Age'].iloc[0]
        p_diam = pipe_table.loc[pipe_table['ID'] == p_id, 'D'].iloc[0]
        p_CP = (-0.0005 * p_diam**2 + 1.9739 * p_diam)*1000
        p_Cr = 1.3*(p_diam/304.8)**0.62*800
        p_len = pipe_table.loc[pipe_table['ID'] == p_id, 'USER_L'].iloc[0]
        
        p_lcc = single_opt_lcc(p_id, p_age, p_diam, p_CP, p_Cr, time_span, t_rep)*p_len/1000
        
        lcc_tot = lcc_tot + p_lcc
        
        t_rep_count =  t_rep_count + 1
        
    return  lcc_tot

#Create function to calculate the single pipe life cycle costs data frame when pipe is replaced at time t_rep
def single_opt_age_rep(pipe_id, pipe_age, pipe_diam, CP, Cr, time_span, t_rep):
    #create empty data frame containing the necessary columns
    p_df = pd.DataFrame(index=range(1, time_span + 1), 
                                    columns=['CI', 'Age', 'Fr', 
                                             'SFr/t', 'CR', 'LCC'])
    #t_rep = single_opt_age(pipe_id, pipe_age, pipe_diam, CP, Cr, time_span)[0]
    #fill the columns with the respective values
    for i in range(1, time_span +1):
        if i <= t_rep:
            p_df.loc[i, 'Age'] = pipe_age + i
            p_df.loc[i, 'CI'] = CP/p_df.loc[i, 'Age']
            p_df.loc[i, 'Fr'] = 0.109 * math.e ** \
                                (-0.0064*pipe_diam) * \
                                p_df.loc[i, 'Age'] ** 1.377
        else:
            p_df.loc[i, 'Age'] = i - t_rep
            p_df.loc[i, 'CI'] = CP/(i - t_rep)
            p_df.loc[i, 'Fr'] = 0.109 * math.e ** \
                                (-0.0064*pipe_diam) * \
                                p_df.loc[i, 'Age'] ** 1.377
        p_df.loc[i, 'SFr/t'] = p_df['Fr'].loc[:i].sum()/i
        p_df.loc[i, 'CR'] = p_df.loc[i, 'SFr/t'] * Cr
        p_df.loc[i, 'LCC'] = p_df.loc[i, 'CI'] + p_df.loc[i, 'CR']
    #return cost data frame
    return p_df

# Calculate invenstment timeseries when an array containing the single pipe replacement time is given:

repl_time_array = np.repeat(10, len(pipes_gdf_cell['ID']))


def investment_series(repl_time_array, pipe_table, p_span):
    # repl_time_array: array containing the replacement time for each pipe
    # pipe_table: data frame containing the pipe data
    lcc_table = pd.DataFrame(index=range(1, p_span + 1), 
                                    columns=pipe_table['ID'],
                                    data = np.nan)
    t_rep_count = 0
    for p_id in pipe_table['ID']:
        t_rep = repl_time_array[t_rep_count]
        p_age = pipe_table.loc[pipe_table['ID'] == p_id, 'Pipe Age'].iloc[0]
        p_diam = pipe_table.loc[pipe_table['ID'] == p_id, 'D'].iloc[0]
        p_CP = (-0.0005 * p_diam**2 + 1.9739 * p_diam)*1000
        p_Cr = 1.3*(p_diam/304.8)**0.62*800   
        
        p_df = single_opt_age_rep(p_id, p_age, p_diam, p_CP, p_Cr, p_span, t_rep)
        
        lcc_table[p_id] = p_df['LCC']
        
        t_rep_count =  t_rep_count + 1
        
    lcc_series = lcc_table.sum(axis=1) 
    
    return lcc_series, lcc_table
    

#find optimal replacement age of all pipes treated as single assets
pipe_table_trep = deepcopy(pipes_gdf_cell)
pipe_table_trep['t_rep'] = np.nan   
pipe_table_trep['LCC_min'] = np.nan 

p_span = 20 # years

# Create a data frame containing the ideal optimal replacement age and minimum LCC of all pipes

for i in range(0,len(pipe_table_trep.index)):
    p_id = pipe_table_trep.loc[i, 'ID']
    p_age = pipe_table_trep.loc[pipe_table_trep['ID'] == p_id, 'Pipe Age'].iloc[0]
    p_diam = pipe_table_trep.loc[pipes_gdf_cell['ID'] == p_id, 'D'].iloc[0]
    p_CP = (0.000005 * p_diam**3 - 0.0033 * p_diam**2 + 0.7188 * p_diam)*1000
    p_Cr = 1.3*(p_diam/304.8)**0.62*800
    
    t_res = single_opt_age(p_id, p_age, p_diam, p_CP, p_Cr, p_span)
    pipe_table_trep.loc[i, 't_rep'] = t_res[0]
    pipe_table_trep.loc[i, 'LCC_min'] = t_res[1]
    
# Calculate the least life cycle cost of the network
pipe_table_trep['LCCmultL'] = pipe_table_trep['USER_L']/1000 * pipe_table_trep['LCC_min']

LLCCn = pipe_table_trep['LCCmultL'].sum()

# Set up the upper and lower boundaries of each variable 

x_base = pipe_table_trep['t_rep'].to_numpy() 

# allowable time span relaxation
a_rel = 3

xl = x_base - a_rel
xl[xl <= 0] = 2

xu = x_base + a_rel
xu[xu > p_span] = p_span

# define the optimization problem
class MyProblem(ElementwiseProblem):

    def __init__(self):
        super().__init__(n_var=len(pipe_table_trep.index),
                         n_obj=2,
                         xl=xl,
                         xu=xu)

    def _evaluate(self, x, out, *args, **kwargs):
        
        lcc_n = lcc_tot_net(x, pipe_table_trep,20)
        f1 = lcc_n - LLCCn
        
        inv_series = investment_series(x, pipe_table_trep,20)
        f2 = inv_series[0].std()
        
        out["F"] = [f1, f2]
        
problem = MyProblem()

algorithm = NSGA2(
    pop_size= 20,
    n_offsprings=16 ,
    sampling=IntegerRandomSampling(),
    crossover=SBX(prob=0.9, eta=15, repair=RoundingRepair()),
    mutation=PM(eta=20, repair=RoundingRepair()),
    eliminate_duplicates=True
)

# execute the optimization
res = minimize(problem,
               algorithm,
               seed=1,
               termination=('n_gen', 1000),
               save_history=True,
               verbose=True)

X = res.X
F = res.F