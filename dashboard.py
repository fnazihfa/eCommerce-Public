# Library
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import geopandas as gpd
import streamlit as st
from babel.numbers import format_currency
sns.set(style='dark')

# Function
def create_daily_orders_df(df):
    daily_orders_df = df.resample(rule='D', on='order_purchase_timestamp').agg({
        "order_id": "nunique",
        "payment_value": "sum"
    })
    daily_orders_df = daily_orders_df.reset_index()
    daily_orders_df.rename(columns={
        "order_id": "order_count",
        "payment_value": "revenue"
    }, inplace=True)
    return daily_orders_df

def create_sum_order_items_df(df):
    sum_order_items_df = df.groupby(by="product_category_name").agg({
        "order_id": "count"
    }).sort_values(by="order_id", ascending=False).reset_index()
    return sum_order_items_df

def create_order_status_df(df):
    count_order_status_df = df.groupby(by="order_status").order_id.nunique().reset_index()
    category_order = ["invoiced", "approved", "processing", "shipped", "delivered", "canceled", "unavailable"]
    all_status_df = pd.DataFrame({"order_status": category_order, "order_id": [0] * len(category_order)})
    
    count_order_status_df = pd.merge(all_status_df, count_order_status_df, on="order_status", how="left")

    count_order_status_df["order_id"] = count_order_status_df["order_id_y"].fillna(0).astype(int)
    count_order_status_df = count_order_status_df[["order_status", "order_id"]]

    count_order_status_df["order_status"] = pd.Categorical(count_order_status_df["order_status"], categories=category_order, ordered=True)
    count_order_status_df = count_order_status_df.sort_values("order_status")
    
    return count_order_status_df

def create_top_cutomers_bystate_df(df, top_n=5):
    customer_bystate_df = df.groupby(by="customer_state").customer_id.nunique().reset_index()
    customer_bystate_df.rename(columns={
        "customer_id": "customer_count"
    }, inplace=True)

    top_customers_bystate_df = customer_bystate_df.nlargest(top_n, 'customer_count')
    return top_customers_bystate_df

def create_top_sellers_bystate_df(df, top_n=5):
    seller_bystate_df = df.groupby(by="seller_state").seller_id.nunique().reset_index()
    seller_bystate_df.rename(columns={
        "seller_id": "seller_count"
    }, inplace=True)

    top_sellers_bystate_df = seller_bystate_df.nlargest(top_n, 'seller_count')
    return top_sellers_bystate_df

def create_map_sellers_customers_df(df):
    world_shapefile = "110m_cultural/ne_110m_admin_0_countries.shp"
    world = gpd.read_file(world_shapefile)

    geolocation_customers_df = df.sort_values(by=['customer_id', 'order_purchase_timestamp'], ascending=[True, False])
    latest_customer_data = geolocation_customers_df.drop_duplicates(subset='customer_id', keep='first')

    geolocation_sellers_df = df.sort_values(by=['seller_id', 'order_purchase_timestamp'], ascending=[True, False])
    latest_seller_data = geolocation_sellers_df.drop_duplicates(subset='seller_id', keep='first')

    customer_map = gpd.GeoDataFrame(
        latest_customer_data,
        geometry=gpd.points_from_xy(latest_customer_data['geolocation_lng'], latest_customer_data['geolocation_lat'])
    )

    seller_map = gpd.GeoDataFrame(
        latest_seller_data,
        geometry=gpd.points_from_xy(latest_seller_data['geolocation_lng'], latest_seller_data['geolocation_lat'])
    )

    return customer_map, seller_map, world

def create_top_states_of_sellers_byorders(df, top_n=5):
    seller_byorders_df = df.groupby("seller_state").order_id.nunique().reset_index()
    seller_byorders_df.rename(columns={
        "order_id": "order_count"
    }, inplace=True)

    top_states_seller_byorders_df = seller_byorders_df.nlargest(top_n, 'order_count')
    return top_states_seller_byorders_df

def create_rfm_df(df):
    rfm_df = df.groupby(by="customer_id", as_index=False).agg({
        "order_purchase_timestamp": "max",
        "order_id": "nunique",
        "payment_value": "sum"
    })
    rfm_df.columns = ["customer_id", "max_order_timestamp", "frequency", "monetary"]

    recent_date = df["order_purchase_timestamp"].max()
    rfm_df["recency"] = rfm_df["max_order_timestamp"].apply(lambda x: (recent_date - x).days)
    rfm_df.drop("max_order_timestamp", axis=1, inplace=True)

    rfm_df["r_rank"] = rfm_df["recency"].rank(ascending=False)
    rfm_df["f_rank"] = rfm_df["frequency"].rank(ascending=True)
    rfm_df["m_rank"] = rfm_df["monetary"].rank(ascending=True)
    
    rfm_df["RFM_score"] = rfm_df[["r_rank", "f_rank", "m_rank"]].mean(axis=1)

    return rfm_df

def create_customer_segment_df(rfm_df):
    rfm_df['r_rank_norm'] = (rfm_df['r_rank'] / rfm_df['r_rank'].max()) * 100
    rfm_df['f_rank_norm'] = (rfm_df['f_rank'] / rfm_df['f_rank'].max()) * 100
    rfm_df['m_rank_norm'] = (rfm_df['m_rank'] / rfm_df['m_rank'].max()) * 100

    rfm_df.drop(columns=['r_rank', 'f_rank', 'm_rank'], inplace=True)

    rfm_df['RFM_score'] = 0.15*rfm_df['r_rank_norm']+0.28 * \
    rfm_df['f_rank_norm']+0.57*rfm_df['m_rank_norm']
    rfm_df['RFM_score'] *= 0.05
    rfm_df = rfm_df.round(2)

    rfm_df["customer_segment"] = np.where(
    rfm_df['RFM_score'] > 4.5, "Top customers", (np.where(
        rfm_df['RFM_score'] > 4, "High value customer",(np.where(
            rfm_df['RFM_score'] > 3, "Medium value customer", np.where(
                rfm_df['RFM_score'] > 1.6, 'Low value customers', 'lost customers'))))))

    customer_segment_df = rfm_df.groupby(by="customer_segment", as_index=False).customer_id.nunique()
    customer_segment_df['customer_segment'] = pd.Categorical(customer_segment_df['customer_segment'], [
        "lost customers", "Low value customers", "Medium value customer", 
        "High value customer", "Top customers"
    ])
    return customer_segment_df

# Load data
all_df = pd.read_csv("all_data.csv")

datetime_columns = ["order_purchase_timestamp", "order_estimated_delivery_date", "shipping_limit_date", "review_creation_date", "review_answer_timestamp"]
all_df.sort_values(by="order_purchase_timestamp", inplace=True)
all_df.reset_index(inplace=True)
 
for column in datetime_columns:
    all_df[column] = pd.to_datetime(all_df[column])

# Filter data
min_date = all_df["order_purchase_timestamp"].min()
max_date = all_df["order_purchase_timestamp"].max()
 
with st.sidebar:
    st.image("Asset/logo.png")

    start_date, end_date = st.date_input(
        label='Time Range :calendar:',
        min_value=min_date,
        max_value=max_date,
        value=[min_date, max_date]
    )

main_df = all_df[(all_df["order_purchase_timestamp"] >= str(start_date)) & 
                (all_df["order_purchase_timestamp"] <= str(end_date))]

# Menyiapkan berbagai dataframe
daily_orders_df = create_daily_orders_df(main_df)
sum_order_items_df = create_sum_order_items_df(main_df)
count_order_status_df = create_order_status_df(main_df)
top_customers_bystate_df = create_top_cutomers_bystate_df(main_df)
top_sellers_bystate_df = create_top_sellers_bystate_df(main_df)
customer_map, seller_map, world = create_map_sellers_customers_df(main_df)
top_states_seller_byorders_df = create_top_states_of_sellers_byorders(main_df)
rfm_df = create_rfm_df(main_df)
customer_segment_df = create_customer_segment_df(rfm_df)

# Plot number of daily orders
st.header('E-Commerce Public Dashboard :sparkles:')
st.subheader('Daily Orders')

col1, col2 = st.columns(2)

with col1:
    total_orders = daily_orders_df.order_count.sum()
    st.metric("Total orders", value=total_orders)

with col2:
    total_revenue = format_currency(daily_orders_df.revenue.sum(), "BRL", locale='es_CO') 
    st.metric("Total Revenue", value=total_revenue)

fig, ax = plt.subplots(figsize=(16, 8))
ax.plot(
    daily_orders_df["order_purchase_timestamp"],
    daily_orders_df["order_count"],
    marker='o', 
    linewidth=2,
    color="#90CAF9"
)
ax.tick_params(axis='y', labelsize=20)
ax.tick_params(axis='x', labelsize=15)

st.pyplot(fig)

# Product performance
st.subheader("Best & Worst Performing Product")
 
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(35, 15))
 
colors = ["#90CAF9", "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3"]
 
sns.barplot(x="order_id", y="product_category_name", data=sum_order_items_df.head(5), hue="product_category_name", palette=colors, legend=False, ax=ax[0])
ax[0].set_ylabel(None)
ax[0].set_xlabel("Number of Sales", fontsize=30)
ax[0].set_title("Best Performing Product", loc="center", fontsize=50)
ax[0].tick_params(axis='y', labelsize=35)
ax[0].tick_params(axis='x', labelsize=30)
 
sns.barplot(x="order_id", y="product_category_name", data=sum_order_items_df.sort_values(by="order_id", ascending=True).head(5), hue="product_category_name", palette=colors, legend=False, ax=ax[1])
ax[1].set_ylabel(None)
ax[1].set_xlabel("Number of Sales", fontsize=30)
ax[1].invert_xaxis()
ax[1].yaxis.set_label_position("right")
ax[1].yaxis.tick_right()
ax[1].set_title("Worst Performing Product", loc="center", fontsize=50)
ax[1].tick_params(axis='y', labelsize=35)
ax[1].tick_params(axis='x', labelsize=30)
 
st.pyplot(fig)

# Order status
st.subheader("Order Status")

fig, ax = plt.subplots(figsize=(20,10))

max_count = count_order_status_df["order_id"].max()
max_status = count_order_status_df.loc[count_order_status_df["order_id"] == max_count, "order_status"].tolist()

colors = ["#90CAF9" if status in max_status else "#D3D3D3" for status in count_order_status_df["order_status"]]

sns.barplot(
    x="order_status",
    y="order_id",
    data=count_order_status_df.sort_values("order_status"),
    palette=colors,
    ax=ax
)
ax.set_title("Number of Order Status", loc="center", fontsize=30)
ax.set_ylabel(None)
ax.set_xlabel(None)
ax.tick_params(axis='x', labelsize=20)
ax.tick_params(axis='y', labelsize=15)

st.pyplot(fig)

# Customer & seller demographic
st.subheader("Top 5 Customer & Seller Demographic")
 
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(35, 15))
 
colors = ["#90CAF9"] + ["#D3D3D3"] * 4
 
sns.barplot(x="customer_count", y="customer_state", data=top_customers_bystate_df.sort_values(by="customer_count", ascending=False), hue="customer_state", palette=colors, legend=False, ax=ax[0])
ax[0].set_ylabel(None)
ax[0].set_xlabel(None)
ax[0].set_title("Number of Customer by States", loc="center", fontsize=50)
ax[0].tick_params(axis='y', labelsize=35)
ax[0].tick_params(axis='x', labelsize=30)
 
sns.barplot(x="seller_count", y="seller_state", data=top_sellers_bystate_df.sort_values(by="seller_count", ascending=False), hue="seller_state", palette=colors, legend=False, ax=ax[1])
ax[1].set_ylabel(None)
ax[1].set_xlabel(None)
ax[1].invert_xaxis()
ax[1].yaxis.set_label_position("right")
ax[1].yaxis.tick_right()
ax[1].set_title("Number of Seller by States", loc="center", fontsize=50)
ax[1].tick_params(axis='y', labelsize=35)
ax[1].tick_params(axis='x', labelsize=30)
 
st.pyplot(fig)

# Map
fig, ax = plt.subplots(figsize=(20, 10))
world.plot(ax=ax, color='lightgrey')

sns.scatterplot(
    x=customer_map.geometry.x, 
    y=customer_map.geometry.y,
    color='blue',
    s=20, 
    label='Customers',
    ax=ax
)

sns.scatterplot(
    x=seller_map.geometry.x, 
    y=seller_map.geometry.y,
    color='green',
    s=20, 
    label='Sellers',
    ax=ax
)

# Adding title and axis labels
ax.set_title('Map of Sellers and Customers', fontsize=30)
ax.set_xlabel(None)
ax.set_ylabel(None)
ax.tick_params(axis='x', labelsize=10)
ax.tick_params(axis='y', labelsize=10)

# Tampilkan plot di Streamlit
st.pyplot(fig)

# Top 5 states of sellers with most orders
st.subheader("Top 5 States of Sellers with Most Orders")

fig, ax = plt.subplots(figsize=(20, 10))
colors = ["#90CAF9"] + ["#D3D3D3"] * 4
sns.barplot(
    x="order_count", 
    y="seller_state",
    data=top_states_seller_byorders_df.sort_values(by="order_count", ascending=False),
    palette=colors,
    ax=ax
)
ax.set_title("Number of Order by Seller States", loc="center", fontsize=30)
ax.set_ylabel(None)
ax.set_xlabel(None)
ax.tick_params(axis='y', labelsize=20)
ax.tick_params(axis='x', labelsize=15)

st.pyplot(fig)

# Best Customer Based on RFM Parameters
st.subheader("Best Customer Based on RFM Parameters (customer_id)")

col1, col2, col3 = st.columns(3)

with col1:
    avg_recency = round(rfm_df.recency.mean(), 1)
    st.metric("Average Recency (days)", value=avg_recency)

with col2:
    avg_frequency = round(rfm_df.frequency.mean(), 2)
    st.metric("Average Frequency", value=avg_frequency)

with col3:
    avg_monetary = format_currency(rfm_df.monetary.mean(), "BRL", locale='es_CO') 
    st.metric("Average Monetary", value=avg_monetary)

fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(5, 10))
colors = ["#90CAF9"] * 5

sns.barplot(x="recency", y="customer_id", data=rfm_df.sort_values(by="recency", ascending=True).head(5), palette=colors, ax=ax[0])
ax[0].set_xlabel(None)
ax[0].set_ylabel(None)
ax[0].set_title("By Recency (days)", loc="center", fontsize=10)
ax[0].tick_params(axis='y', labelsize=10)
ax[0].tick_params(axis='x', labelsize=10)
 
sns.barplot(x="frequency", y="customer_id", data=rfm_df.sort_values(by="frequency", ascending=False).head(5), palette=colors, ax=ax[1])
ax[1].set_xlabel(None)
ax[1].set_ylabel(None)
ax[1].set_title("By Frequency", loc="center", fontsize=10)
ax[1].tick_params(axis='y', labelsize=10)
ax[1].tick_params(axis='x', labelsize=10)
 
sns.barplot(x="monetary", y="customer_id", data=rfm_df.sort_values(by="monetary", ascending=False).head(5), palette=colors, ax=ax[2])
ax[2].set_xlabel(None)
ax[2].set_ylabel(None)
ax[2].set_title("By Monetary", loc="center", fontsize=10)
ax[2].tick_params(axis='y', labelsize=10)
ax[2].tick_params(axis='x', labelsize=10)
 
st.pyplot(fig)

# Customer Segment
st.subheader("Customer Segment")

fig, ax = plt.subplots(figsize=(10,5))
colors = ["#90CAF9"] * 2 + ["#D3D3D3"] * 3

sns.barplot(
    x="customer_id",
    y="customer_segment",
    data=customer_segment_df.sort_values(by="customer_segment", ascending=False),
    palette=colors,
)
ax.set_title("Number of Customer for Each Segment", loc="center", fontsize=15)
ax.set_ylabel(None)
ax.set_xlabel(None)
ax.tick_params(axis='y', labelsize=10)
ax.tick_params(axis='x', labelsize=10)

st.pyplot(fig)

st.caption('Copyright © fadiyahnazihfa 2024')