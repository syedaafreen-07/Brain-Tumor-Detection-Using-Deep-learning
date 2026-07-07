def show_model_performance():
    import streamlit as st
    import joblib
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from styles import style_graph

    @st.cache_data
    def load_data():
        return joblib.load("model_performance.pkl")
    res=load_data()

    st.subheader("📊 Model Performance Dashboard")

    model_names = [m for m in res.keys() if m != "summary"]
    model_name = st.selectbox("🔍 Select Model", model_names)
    model_data = res[model_name]

   
    st.markdown("## 📌 Metrics Overview")
    col1, col2, col3, col4 = st.columns(4)

    report = model_data["classification_report"]
    df_report = pd.DataFrame(report).transpose()
    precision = df_report.loc["weighted avg", "precision"]
    recall = df_report.loc["weighted avg", "recall"]
    f1 = df_report.loc["weighted avg", "f1-score"]
    accuracy = res["summary"][model_name]["accuracy"]
    col1.metric("Accuracy", f"{accuracy:.4f}")
    col2.metric("Precision", f"{precision:.4f}")
    col3.metric("Recall", f"{recall:.4f}")
    col4.metric("F1 Score", f"{f1:.4f}")

    if model_name == "Xception":
        xception_his= {
        "accuracy": [
            0.7513, 0.8788, 0.9173, 0.9502, 0.9661,0.9786, 0.9864, 0.9936
        ],
        "loss": [
            0.6716, 0.3607, 0.2504, 0.1655, 0.1195,0.0811, 0.0618, 0.0383
        ]
    }
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Training Accuracy Curve")
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.plot(
                xception_his["accuracy"],
                marker="o",
                linewidth=2,
                color="#6FB0E2"
            )
            style_graph(ax, fig, "Training Accuracy")
            st.pyplot(fig)

        with col2:
            st.subheader("Training Loss Curve")
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.plot(
                xception_his["loss"],
                marker="o",
                linewidth=3,
                color="#CDB4F6"
            )
            style_graph(ax, fig, "Training Loss")
            st.pyplot(fig)
    
    st.markdown("## 🏆 Model Comparison Table")
    summary_df = pd.DataFrame(res["summary"]).T
    summary_df = summary_df.reset_index().rename(columns={"index": "Model"})
    styled_df = summary_df.style \
    .set_table_styles([
        {
            'selector': 'th',
            'props': [
                ('background-color', "#ADD8F7"),
                ('color', "#2b5e83"),
                ('text-align', 'center')
            ]
        }
    ]) \
    .set_properties(**{
        'background-color': '#ADD8F7',
        'color': "#18415f",
        'text-align':'center'
    })
    st.table(styled_df)
        
    fig, ax = plt.subplots(figsize=(4, 2))
    sns.barplot(data=summary_df, x="Model", y="accuracy", palette=["#6FB0E2","#A8D5F2","#CDB4F6","#89CFF0"],ax=ax)
    plt.xticks(rotation=45)
    style_graph(ax, fig, "Model Comparison")
    st.pyplot(fig)

    st.markdown("## 📊 Confusion Matrix")
    cm = model_data["confusion_matrix"]
    fig, ax = plt.subplots(figsize=(3, 2))
    sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="PuBu",
    linewidths=1,
    linecolor="white",
    ax=ax)
    style_graph(ax, fig, "Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    st.markdown("## 📄 Classification Report")

    df_report = df_report.reset_index().rename(columns={"index": "Model"})
    styled_df = df_report.style \
    .set_table_styles([
        {
            'selector': 'th',
            'props': [
                ('background-color', '#CFE8FA'),
                ('color', "#19405c"),
                ('text-align', 'center')
            ]
        }
    ]) \
    .set_properties(**{
        'background-color': '#CFE8FA',
        'color': "#2b5e83",
        'text-align':'center'
    })
    st.table(styled_df)

    st.markdown("## 📈 ROC Curve")
    roc = model_data.get("roc", None)
    if roc:
        fig, ax = plt.subplots(figsize=(4,2))
        ax.plot(
            roc["fpr"],
            roc["tpr"],
            linewidth=3,
            color="#6FB0E2",
            label=f"AUC = {roc['auc']:.3f}")
        ax.plot([0,1],[0,1], linestyle="--", color="#CDB4F6")
        style_graph(ax, fig, "ROC Curve")
        ax.plot([0, 1], [0, 1], linestyle="--")
        ax.set_title(f"ROC Curve - {model_name}")
        ax.set_xlabel("False Positive Rate")
        
        ax.legend()
        st.pyplot(fig)
    else:
        st.warning("ROC data not available for this model")

   
    best_model = summary_df.loc[summary_df["accuracy"].idxmax(), "Model"]
    st.success(f"🏆 Best Performing Model: {best_model}")