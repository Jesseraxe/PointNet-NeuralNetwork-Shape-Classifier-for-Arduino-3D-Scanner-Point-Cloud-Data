# app.py
import streamlit as st
import torch
import numpy as np
import plotly.graph_objects as go
import os
from shape_classifier import PointNet

def load_model(model_path):
    """Load the trained model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PointNet(num_classes=5).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, checkpoint

def process_point_cloud(file):
    """Process uploaded point cloud file"""
    try:
        # Skip header rows and load points
        points = np.loadtxt(file, delimiter=',', skiprows=3)
        
        # Store original points for visualization
        original_points = points.copy()
        
        # Normalize points for model input
        centroid = np.mean(points, axis=0)
        points -= centroid
        dist = np.max(np.sqrt(np.sum(points ** 2, axis=1)))
        if dist > 0:
            points /= dist
        
        # Sample 1024 points if necessary
        if len(points) > 1024:
            indices = np.random.choice(len(points), 1024, replace=False)
            points = points[indices]
        elif len(points) < 1024:
            indices = np.random.choice(len(points), 1024, replace=True)
            points = points[indices]
        
        return points, original_points
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return None, None

def visualize_point_cloud(points):
    """Create a 3D scatter plot of the point cloud using Plotly"""
    fig = go.Figure(data=[go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode='markers',
        marker=dict(
            size=2,
            color=points[:, 2],
            colorscale='Viridis',
            opacity=0.8
        )
    )])
    
    fig.update_layout(
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode='data'
        ),
        width=700,
        height=700,
        margin=dict(l=0, r=0, b=0, t=30)
    )
    
    return fig

def main():
    st.title("3D Shape Classifier")
    st.write("Upload a point cloud file to classify the 3D shape")

    # File uploader
    uploaded_file = st.file_uploader("Choose a point cloud file", type=['txt'])
    
    if uploaded_file is not None:
        # Process the point cloud
        points, original_points = process_point_cloud(uploaded_file)
        
        if points is not None and original_points is not None:
            # Display point cloud visualization
            st.subheader("Point Cloud Visualization")
            fig = visualize_point_cloud(original_points)
            st.plotly_chart(fig)
            
            # Load model
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model, _ = load_model('assets/model/best_model.pth')
            model.eval()
            
            # Prepare input
            points_tensor = torch.from_numpy(points).float().unsqueeze(0)
            points_tensor = points_tensor.transpose(2, 1).to(device)
            
            # Make prediction
            with torch.no_grad():
                outputs = model(points_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                predicted_class = outputs.max(1)[1].item()
            
            # Display results
            class_names = ['sphere', 'cone', 'cube', 'cylinder', 'cuboid']
            st.subheader("Classification Result")
            st.write(f"Predicted Shape: {class_names[predicted_class]}")
            
            # Display point cloud statistics
            st.subheader("Point Cloud Information")
            st.write(f"Number of points: {len(original_points)}")
            st.write(f"Point cloud dimensions: {original_points.shape}")
            st.write(f"Number of points used for classification: {len(points)}")

if __name__ == "__main__":
    main()