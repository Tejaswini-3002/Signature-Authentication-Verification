import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from plotly.subplots import make_subplots


class DataVisualizer:
    def __init__(self):
        self.layout = {
            'template': 'plotly_white',
            'plot_bgcolor': 'white',
            'paper_bgcolor': 'black',
            'font': dict(color='white')
        }
    
    def plot_correlation_heatmap(self, data):
        """
        Create a correlation heatmap for numerical features using Plotly
        """
        data["Gender"] = data["Gender"].map({"Male": 1, "Female": 0})
        correlation_matrix = data.corr()

        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.columns,
            colorscale='RdBu',
            zmin=-1,
            zmax=1,
            colorbar=dict(title='Correlation'),
            text=correlation_matrix.round(2).astype(str).values, 
            texttemplate="%{text}",  
            hoverinfo="text" 
        ))

        fig.update_layout(
            **self.layout,
            title='Correlation Heatmap',
            title_x=0.4,
            xaxis_title='Features',
            yaxis_title='Features',
            width=1000,
            height=800,
            margin=dict(l=50, r=50, t=50, b=50)
        )
        return fig
    
    def plot_boxplots(self, data):
        """
        Create boxplots for numerical features, arranged in rows of 3.
        """
        numerical_columns = ['Age', 'Total Bilirubin', 'Direct Bilirubin',
           'Alkaline Phosphotase', 'Sgpt Alamine Aminotransferase',
           'Sgot Aspartate Aminotransferase', 'Total Proteins', 'ALB Albumin',
           'A/G ratio Albumin and Globulin Ratio']
        
        n_cols = 3
        n_rows = (len(numerical_columns) // n_cols) + (1 if len(numerical_columns) % n_cols != 0 else 0)
        
        fig = make_subplots(rows=n_rows, cols=n_cols, horizontal_spacing=0.1, vertical_spacing=0.08)
        
        for idx, col in enumerate(numerical_columns):
            row = (idx // n_cols) + 1
            col_num = (idx % n_cols) + 1
            fig.add_trace(go.Box(y=data[col], name=col, boxmean=True), row=row, col=col_num)
        
        fig.update_layout(
            **self.layout,
            title='Boxplots of Numerical Features',
            title_x=0.3, 
            showlegend=False,
            width=700 * n_cols,
            height=500 * n_rows,
            margin=dict(t=50, b=50, l=50, r=50)
        )
        return fig

    def plot_histograms(self, data):
        """
        Create continuous-looking histograms for numerical features with 'Result' as hue
        Three plots per row
        """
        numerical_columns = ['Age', 'Total Bilirubin', 'Direct Bilirubin',
                            'Alkaline Phosphotase', 'Sgpt Alamine Aminotransferase',
                            'Sgot Aspartate Aminotransferase', 'Total Proteins', 
                            'ALB Albumin', 'A/G ratio Albumin and Globulin Ratio']
        
        n_cols = 3  # Changed to 3 plots per row
        n_rows = (len(numerical_columns) // n_cols) + (1 if len(numerical_columns) % n_cols != 0 else 0)
        
        fig = make_subplots(rows=n_rows, cols=n_cols, 
                        vertical_spacing=0.1,
                        horizontal_spacing=0.1)  # Slightly reduced horizontal spacing
        
        for idx, col in enumerate(numerical_columns):
            row = (idx // n_cols) + 1
            col_num = (idx % n_cols) + 1
            
            # Calculate optimal number of bins using Freedman-Diaconis rule
            data_range = data[col].max() - data[col].min()
            iqr = data[col].quantile(0.75) - data[col].quantile(0.25)
            bin_width = 2 * iqr / (len(data[col]) ** (1/3))
            n_bins = max(int(data_range / bin_width), 30)  # At least 30 bins for smoothness
            
            # Create separate histograms for each Result category
            for result in sorted(data['Result'].unique()):
                subset = data[data['Result'] == result]
                
                hist_trace = go.Histogram(
                    x=subset[col],
                    name=f'Result {result}',
                    nbinsx=n_bins,
                    opacity=0.7,
                    marker_color='blue' if result == 1 else 'orange',
                    showlegend=(idx == 0)  # Show legend only for first subplot
                )
                fig.add_trace(hist_trace, row=row, col=col_num)
            
            # Update layout for each subplot
            fig.update_xaxes(title_text=col, row=row, col=col_num, title_standoff=5)
            fig.update_yaxes(title_text="Count", row=row, col=col_num, title_standoff=5)

        # Update overall layout with adjusted width for 3 columns
        fig.update_layout(
            **self.layout,
            title_text='Feature Distributions by Result',
            title_x=0.3,
            barmode='overlay',
            showlegend=True,
            legend=dict(
                title='Result',
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            height=400 * n_rows,  # Slightly reduced height per row
            width=700 * n_cols,  # Increased width to accommodate 3 columns
            margin=dict(t=120, b=50, l=50, r=50)
        )
        
        return fig
    
    def plot_pairplot(self, data):
        """
        Create a scatter matrix (pairplot) for numerical features colored by 'Result'.
        """
        numerical_features = ['Age', 'Total Bilirubin', 'Direct Bilirubin', 
            'Alkaline Phosphotase', 'Sgpt Alamine Aminotransferase', 
            'Sgot Aspartate Aminotransferase', 'Total Proteins', 
            'ALB Albumin', 'A/G ratio Albumin and Globulin Ratio']
        data["Result"] = data["Result"].astype(str)
        
        fig = px.scatter_matrix(
            data, 
            dimensions=numerical_features,  
            color="Result", 
            color_discrete_map={"1": "blue", "2": "orange"}, 
            title="Pairplot of Numerical Features",
            width=2250,  
            height=2250
        )
        
        fig.update_layout(
            **self.layout,
            title="Pairplot of Numerical Features",
            title_x=0.4,
            margin=dict(l=10, r=10, b=50, t=50),

        )
        return fig