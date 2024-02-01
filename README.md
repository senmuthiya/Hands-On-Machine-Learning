{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyOlNZEFLppHG7UK2aPlxDvI",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/senmuthiya/Hands-On-Machine-Learning/blob/Dev/Data_Science_Essentials_Building_Robust_Models_with_scikit_learn.ipynb\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "### Below is a step-by-step guide to creating a supervised learning model in Python using Google Colab sample datasets of california_housing_train.csv and california_housing_test.csv.\n",
        "\n",
        "In the first part we will use scikit-learn to build the model and later we will use LazyPredict Python library that automates the model selection and evaluation process."
      ],
      "metadata": {
        "id": "2MzKi_L5ViuS"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Import Libraries\n",
        "---\n",
        "Import the necessary libraries for your machine learning task. In this case, we'll use pandas for data manipulation, scikit-learn for building the model, and matplotlib, seaborn for visualization."
      ],
      "metadata": {
        "id": "HroUCkafQQcZ"
      }
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "bCswo_W9Lnzr"
      },
      "outputs": [],
      "source": [
        "import pandas as pd\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.linear_model import LinearRegression\n",
        "from sklearn.metrics import mean_squared_error\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Load and Explore Data:\n",
        "---\n",
        "Load your training and testing datasets using pandas.\n",
        "\n"
      ],
      "metadata": {
        "id": "1moYu07AS9yB"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# Load training dataset\n",
        "train_data = pd.read_csv('/content/sample_data/california_housing_train.csv')\n",
        "\n",
        "# Load testing dataset\n",
        "test_data = pd.read_csv('/content/sample_data/california_housing_test.csv')\n",
        "\n",
        "# Explore the dat\n",
        "print(train_data.head())"
      ],
      "metadata": {
        "id": "1D8tza3ZRRgW",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "fd258b7c-f57d-4df6-b44f-ba1be79aa813"
      },
      "execution_count": 2,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "   longitude  latitude  housing_median_age  total_rooms  total_bedrooms  \\\n",
            "0    -114.31     34.19                15.0       5612.0          1283.0   \n",
            "1    -114.47     34.40                19.0       7650.0          1901.0   \n",
            "2    -114.56     33.69                17.0        720.0           174.0   \n",
            "3    -114.57     33.64                14.0       1501.0           337.0   \n",
            "4    -114.57     33.57                20.0       1454.0           326.0   \n",
            "\n",
            "   population  households  median_income  median_house_value  \n",
            "0      1015.0       472.0         1.4936             66900.0  \n",
            "1      1129.0       463.0         1.8200             80100.0  \n",
            "2       333.0       117.0         1.6509             85700.0  \n",
            "3       515.0       226.0         3.1917             73400.0  \n",
            "4       624.0       262.0         1.9250             65500.0  \n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Data Preprocessing\n",
        "---\n",
        "Handle missing values, encode categorical variables (if any), and perform other necessary preprocessing."
      ],
      "metadata": {
        "id": "-o0NL-VeTsgj"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# Check for missing values\n",
        "print(train_data.isnull().sum())\n",
        "\n",
        "# Handle missing values (if needed)\n",
        "train_data = train_data.dropna()\n",
        "\n",
        "# Encode categorical variables (if needed)\n",
        "# For example, you can use pd.get_dummies() or sklearn's LabelEncoder\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "2a04IngSTRn1",
        "outputId": "3db3dddc-9af5-4c11-d47f-adb0f172f901"
      },
      "execution_count": 3,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "longitude             0\n",
            "latitude              0\n",
            "housing_median_age    0\n",
            "total_rooms           0\n",
            "total_bedrooms        0\n",
            "population            0\n",
            "households            0\n",
            "median_income         0\n",
            "median_house_value    0\n",
            "dtype: int64\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Split Data into Features and Target\n",
        "---\n",
        "Define the features (X) and the target variable (y)."
      ],
      "metadata": {
        "id": "Utn8YTvUUD1N"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "X_train = train_data.drop('median_house_value', axis=1)\n",
        "y_train = train_data['median_house_value']\n",
        "\n",
        "X_test = test_data.drop('median_house_value', axis=1)\n",
        "y_test = test_data['median_house_value']"
      ],
      "metadata": {
        "id": "-k3K83IuUPAk"
      },
      "execution_count": 4,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Train-Test Split\n",
        "---\n",
        "Split the training data into training and validation sets. (in our case we dont have to do this as we already have seperate data sets)"
      ],
      "metadata": {
        "id": "zZT2mtAHfn0s"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# use this code to Train-Test Split\n",
        "# X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)\n"
      ],
      "metadata": {
        "id": "Ly_Lzy-Nf4UO"
      },
      "execution_count": 6,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Train the Model\n",
        "---\n",
        "Create and train the supervised learning model. In this example, I'm using linear regression."
      ],
      "metadata": {
        "id": "EvTBK8Wub7GL"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "model = LinearRegression()\n",
        "model.fit(X_train, y_train)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 74
        },
        "id": "fbHExvmhe0hH",
        "outputId": "83acbf80-db96-4252-9bae-53b66a309fc8"
      },
      "execution_count": 7,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "LinearRegression()"
            ],
            "text/html": [
              "<style>#sk-container-id-2 {color: black;background-color: white;}#sk-container-id-2 pre{padding: 0;}#sk-container-id-2 div.sk-toggleable {background-color: white;}#sk-container-id-2 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-2 label.sk-toggleable__label-arrow:before {content: \"▸\";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-2 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-2 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-2 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-2 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-2 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-2 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: \"▾\";}#sk-container-id-2 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-2 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-2 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-2 div.sk-parallel-item::after {content: \"\";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-2 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 div.sk-serial::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-2 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-2 div.sk-item {position: relative;z-index: 1;}#sk-container-id-2 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-2 div.sk-item::before, #sk-container-id-2 div.sk-parallel-item::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-2 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-2 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-2 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-2 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-2 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-2 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-2 div.sk-label-container {text-align: center;}#sk-container-id-2 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-2 div.sk-text-repr-fallback {display: none;}</style><div id=\"sk-container-id-2\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>LinearRegression()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-2\" type=\"checkbox\" checked><label for=\"sk-estimator-id-2\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">LinearRegression</label><div class=\"sk-toggleable__content\"><pre>LinearRegression()</pre></div></div></div></div></div>"
            ]
          },
          "metadata": {},
          "execution_count": 7
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Make Predictions\n",
        "---\n",
        "Use the trained model to make predictions on the testing dataset."
      ],
      "metadata": {
        "id": "GuBxKHHSfB11"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "y_pred = model.predict(X_test)\n"
      ],
      "metadata": {
        "id": "qBPebQlVgR0_"
      },
      "execution_count": 8,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Evaluate the Model\n",
        "\n",
        "Evaluate the performance of the model using appropriate metrics."
      ],
      "metadata": {
        "id": "lZMyJ2magfY1"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "mse = mean_squared_error(y_test, y_pred)\n",
        "print(f'Mean Squared Error: {mse}')\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "kyHVNiqSg18e",
        "outputId": "387151e9-5872-43c1-b1fd-a6d25ee860b6"
      },
      "execution_count": 9,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Mean Squared Error: 4867205486.928806\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Visualize Results\n",
        "---\n",
        "Visualize the predicted values against the actual values to get an understanding of how well the model is performing."
      ],
      "metadata": {
        "id": "TGyeC8IGg_3T"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "plt.scatter(y_test, y_pred)\n",
        "plt.xlabel('Actual Values')\n",
        "plt.ylabel('Predicted Values')\n",
        "plt.title('Actual vs Predicted Values')\n",
        "plt.show()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 472
        },
        "id": "RDkBELrphnzR",
        "outputId": "24c488f7-7b67-405b-98b6-44d7fca617ba"
      },
      "execution_count": 10,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 640x480 with 1 Axes>"
            ],
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAAmUAAAHHCAYAAAD+sy9fAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/bCgiHAAAACXBIWXMAAA9hAAAPYQGoP6dpAACOxklEQVR4nO3deVxUVf8H8M8MOwgDiAiu4JJKuK+kqSmFSYtppaZp5qPZT8u0RX0ql6wsfUrt0bTVesq9NE2Ncl9xFxVxJVxScAEBQVnn/v6gO80Ms9w7+8Dn/XrxKmbO3Hu4zvKdc77nexSCIAggIiIiIqdSOrsDRERERMSgjIiIiMglMCgjIiIicgEMyoiIiIhcAIMyIiIiIhfAoIyIiIjIBTAoIyIiInIBDMqIiIiIXACDMiIiIiIXwKCMiFyCQqHA9OnTnd0Np+vZsyd69uyp+f3ixYtQKBT47rvvnNYnffp9dJQXXngBUVFRDj8vkaMwKCOqgj7//HMoFAp07tzZ4mNcu3YN06dPR0pKiu065uJ27NgBhUKh+fHy8kKjRo0wbNgw/Pnnn87uniz79u3D9OnTkZub6/BzHz16FAqFAu+8847RNufPn4dCocDEiRMd2DMi18agjKgKWrp0KaKionDw4EFcuHDBomNcu3YNM2bMqFZBmejVV1/FDz/8gC+//BKJiYlYuXIlOnbsiGvXrjm8Lw0bNsS9e/fw/PPPy3rcvn37MGPGDKcEZe3atUPz5s2xfPlyo22WLVsGABg6dKijukXk8hiUEVUxGRkZ2LdvHz799FPUqlULS5cudXaX3M6DDz6IoUOHYsSIEfjvf/+L//znP8jJycH3339v9DGFhYV26YtCoYCvry88PDzscnx7GTJkCP7880/s37/f4P3Lly9H8+bN0a5dOwf3jMh1MSgjqmKWLl2KkJAQJCYm4umnnzYalOXm5mLChAmIioqCj48P6tWrh2HDhuHWrVvYsWMHOnbsCAAYMWKEZjpPzGuKiorCCy+8UOmY+rlGJSUlmDp1Ktq3bw+VSoWAgAA8+OCD2L59u+y/6/r16/D09MSMGTMq3Xf27FkoFAosWLAAAFBaWooZM2agadOm8PX1Rc2aNdGtWzds3rxZ9nkBoFevXgAqAl4AmD59OhQKBdLS0vDcc88hJCQE3bp107T/8ccf0b59e/j5+SE0NBSDBg3ClStXKh33yy+/ROPGjeHn54dOnTph9+7dldoYyyk7c+YMnn32WdSqVQt+fn5o1qwZ3n77bU3/3nzzTQBAdHS05t/v4sWLdumjIUOGDAHwz4iYtiNHjuDs2bOaNuvWrUNiYiLq1KkDHx8fNG7cGDNnzkR5ebnJc4jTzTt27NC53dQ1e/rppxEaGgpfX1906NAB69ev12lj6+cOkRwMyoiqmKVLl6J///7w9vbG4MGDcf78eRw6dEinTUFBAR588EH897//xSOPPIL58+djzJgxOHPmDP766y+0aNEC7733HgBg9OjR+OGHH/DDDz+ge/fusvqSn5+Pr7/+Gj179sTHH3+M6dOn4+bNm0hISJA9LVq7dm306NEDq1atqnTfypUr4eHhgWeeeQZARVAyY8YMPPTQQ1iwYAHefvttNGjQAEePHpV1TlF6ejoAoGbNmjq3P/PMM7h79y4+/PBDjBo1CgDwwQcfYNiwYWjatCk+/fRTvPbaa9i6dSu6d++uM5X4zTff4KWXXkJERARmz56Nrl274oknnjAYGOk7ceIEOnfujG3btmHUqFGYP38++vXrh19//RUA0L9/fwwePBgAMHfuXM2/X61atRzWx+joaDzwwANYtWpVpeBKDNSee+45AMB3332HGjVqYOLEiZg/fz7at2+PqVOnYvLkyWbPI9WpU6fQpUsXnD59GpMnT8Ynn3yCgIAA9OvXD2vXrtW0s/Vzh0gWgYiqjMOHDwsAhM2bNwuCIAhqtVqoV6+eMH78eJ12U6dOFQAIa9asqXQMtVotCIIgHDp0SAAgLFmypFKbhg0bCsOHD690e48ePYQePXpofi8rKxOKi4t12ty+fVuoXbu28OKLL+rcDkCYNm2ayb/viy++EAAIJ0+e1Lk9JiZG6NWrl+b31q1bC4mJiSaPZcj27dsFAMK3334r3Lx5U7h27ZqwceNGISoqSlAoFMKhQ4cEQRCEadOmCQCEwYMH6zz+4sWLgoeHh/DBBx/o3H7y5EnB09NTc3tJSYkQHh4utGnTRuf6fPnllwIAnWuYkZFR6d+he/fuQmBgoHDp0iWd84j/doIgCHPmzBEACBkZGXbvozELFy4UAAi///675rby8nKhbt26QlxcnOa2u3fvVnrsSy+9JPj7+wtFRUWa24YPHy40bNhQ87v477V9+3adxxq6Zr179xZatmypczy1Wi088MADQtOmTTW3WfrcIbIFjpQRVSFLly5F7dq18dBDDwGoyEcaOHAgVqxYoTNa8fPPP6N169Z46qmnKh1DoVDYrD8eHh7w9vYGAKjVauTk5KCsrAwdOnSwaOShf//+8PT0xMqVKzW3paamIi0tDQMHDtTcFhwcjFOnTuH8+fMW9fvFF19ErVq1UKdOHSQmJqKwsBDff/89OnTooNNuzJgxOr+vWbMGarUazz77LG7duqX5iYiIQNOmTTXTtocPH8aNGzcwZswYzfUBKko+qFQqk327efMmdu3ahRdffBENGjTQuU/Kv50j+igaOHAgvLy8dKYwd+7ciatXr2qmLgHAz89P8/937tzBrVu38OCDD+Lu3bs4c+aMpHOZkpOTg23btuHZZ5/VHP/WrVvIzs5GQkICzp8/j6tXrwKw/rlDZA0GZURVRHl5OVasWIGHHnoIGRkZuHDhAi5cuIDOnTvj+vXr2Lp1q6Zteno6YmNjHdKv77//Hq1atdLk59SqVQsbN25EXl6e7GOFhYWhd+/eOlOYK1euhKenJ/r376+57b333kNubi7uu+8+tGzZEm+++SZOnDgh+TxTp07F5s2bsW3bNpw4cQLXrl0zuPoxOjpa5/fz589DEAQ0bdoUtWrV0vk5ffo0bty4AQC4dOkSAKBp06Y6jxdLcJgiluaw9N/PEX0U1axZEwkJCVi7di2KiooAVExdenp64tlnn9W0O3XqFJ566imoVCoEBQWhVq1amlWZljxP9F24cAGCIODdd9+t9DdPmzYNADR/t7XPHSJreDq7A0RkG9u2bUNmZiZWrFiBFStWVLp/6dKleOSRR2xyLmMjMuXl5TqrBH/88Ue88MIL6NevH958802Eh4fDw8MDs2bN0uRpyTVo0CCMGDECKSkpaNOmDVatWoXevXsjLCxM06Z79+5IT0/HunXr8Mcff+Drr7/G3LlzsXjxYvzrX/8ye46WLVsiPj7ebDvtER6gYjRQoVDgt99+M7haskaNGhL+QvtydB+HDh2KDRs2YMOGDXjiiSfw888/45FHHtHkt+Xm5qJHjx4ICgrCe++9h8aNG8PX1xdHjx7FpEmToFarjR7b1PNQm3iMN954AwkJCQYf06RJEwDWP3eIrMGgjKiKWLp0KcLDw7Fw4cJK961ZswZr167F4sWL4efnh8aNGyM1NdXk8UxNhYWEhBisf3Xp0iWdUZSffvoJjRo1wpo1a3SOJ45OWKJfv3546aWXNFOY586dw5QpUyq1Cw0NxYgRIzBixAgUFBSge/fumD59ul0/WBs3bgxBEBAdHY377rvPaLuGDRsCqBi1Eld2AhUr/zIyMtC6dWujjxWvr6X/fo7oo7YnnngCgYGBWLZsGby8vHD79m2dqcsdO3YgOzsba9as0VlIIq50NSUkJAQAKj0XxVE+kXjNvLy8JAXbznjuEAGcviSqEu7du4c1a9bgsccew9NPP13pZ9y4cbhz545m+f+AAQNw/PhxnVVnIkEQAAABAQEAKn/gARUf7Pv370dJSYnmtg0bNlRalSeOxIjHBIADBw4gOTnZ4r81ODgYCQkJWLVqFVasWAFvb2/069dPp012drbO7zVq1ECTJk1QXFxs8Xml6N+/Pzw8PDBjxgydvxmouAZivzp06IBatWph8eLFOtfwu+++M1vstVatWujevTu+/fZbXL58udI5RMb+/RzRR21+fn546qmnsGnTJixatAgBAQF48sknNfcbeo6UlJTg888/N3vshg0bwsPDA7t27dK5Xf+x4eHh6NmzJ7744gtkZmZWOs7Nmzc1/++s5w4RwJEyoiph/fr1uHPnDp544gmD93fp0kVTSHbgwIF488038dNPP+GZZ57Biy++iPbt2yMnJwfr16/H4sWL0bp1azRu3BjBwcFYvHgxAgMDERAQgM6dOyM6Ohr/+te/8NNPP6FPnz549tlnkZ6ejh9//BGNGzfWOe9jjz2GNWvW4KmnnkJiYiIyMjKwePFixMTEoKCgwOK/d+DAgRg6dCg+//xzJCQkIDg4WOf+mJgY9OzZE+3bt0doaCgOHz6Mn376CePGjbP4nFI0btwY77//PqZMmYKLFy+iX79+CAwMREZGBtauXYvRo0fjjTfegJeXF95//3289NJL6NWrFwYOHIiMjAwsWbJEUr7WZ599hm7duqFdu3YYPXo0oqOjcfHiRWzcuFFTaqR9+/YAgLfffhuDBg2Cl5cXHn/8cYf1UdvQoUPxv//9D7///juGDBmiCRgB4IEHHkBISAiGDx+OV199FQqFAj/88EOlgNEQlUqFZ555Bv/973+hUCjQuHFjbNiwQZMfpm3hwoXo1q0bWrZsiVGjRqFRo0a4fv06kpOT8ddff+H48eMAnPfcIQLAkhhEVcHjjz8u+Pr6CoWFhUbbvPDCC4KXl5dw69YtQRAEITs7Wxg3bpxQt25dwdvbW6hXr54wfPhwzf2CIAjr1q0TYmJiBE9Pz0olBj755BOhbt26go+Pj9C1a1fh8OHDlUpiqNVq4cMPPxQaNmwo+Pj4CG3bthU2bNhQqbSBIEgriSHKz88X/Pz8BADCjz/+WOn+999/X+jUqZMQHBws+Pn5Cc2bNxc++OADoaSkxORxxRILq1evNtlOLIlx8+ZNg/f//PPPQrdu3YSAgAAhICBAaN68uTB27Fjh7NmzOu0+//xzITo6WvDx8RE6dOgg7Nq1q9I1NFTeQRAEITU1VXjqqaeE4OBgwdfXV2jWrJnw7rvv6rSZOXOmULduXUGpVFYqj2HLPppTVlYmREZGCgCETZs2Vbp/7969QpcuXQQ/Pz+hTp06wltvvSX8/vvvlcpdGHre3Lx5UxgwYIDg7+8vhISECC+99JKQmppq8Jqlp6cLw4YNEyIiIgQvLy+hbt26wmOPPSb89NNPmjaWPneIbEEhCBK+jhARERGRXTGnjIiIiMgFMCgjIiIicgEMyoiIiIhcAIMyIiIiIhfAoIyIiIjIBTAoIyIiInIBLB7rotRqNa5du4bAwECT290QERGR6xAEAXfu3EGdOnWgVMob+2JQ5qKuXbuG+vXrO7sbREREZIErV66gXr16sh7DoMxFBQYGAqj4Rw0KCnJyb4iIiEiK/Px81K9fX/M5LgeDMhclTlkGBQUxKCMiInIzlqQeMdGfiIiIyAUwKCMiIiJyAQzKiIiIiFwAgzIiIiIiF8CgjIiIiMgFMCgjIiIicgEMyoiIiIhcAIMyIiIiIhfAoIyIiIjIBbCiPxEREVUr5WoBBzNycONOEcIDfdEpOhQeSvkV+G2NQRkRERFVG0mpmZjxaxoy84o0t0WqfDHt8Rj0iY10Ys84fUlERETVRFJqJl7+8ahOQAYAWXlFePnHo0hKzXRSzyowKCMiIqIqr1wtYMavaRAM3CfeNuPXNJSrDbVwDAZlREREVOUdzMipNEKmTQCQmVeEgxk5juuUHgZlREREVOXduGM8ILOknT0wKCMiIqIqLzzQ16bt7IFBGREREVV57RuGwFzVC6Wiop2zMCgjIiKiKu/Ipdswl8OvFiraOQuDMiIiIqrymFNGRERE5ALCavjYtJ09MCgjIiKiKk9dLq3+mNR29sCgjIiIiKq85IxbNm1nDwzKiIiIqMq7eltarpjUdvbAoIyIiIiqPIWZchhy29kDgzIiIiKq8uqG+Nm0nT0wKCMiIqIqr0tUTZu2swcGZURERFT1SZ2W5PQlERERkf0cyMixaTt7YFBGREREVZ4AafXHpLazBwZlREREVOUF+3nZtJ09MCgjIiKiKo/bLBERERG5gAiVtFIXUtvZA4MyIiIiqvI6RYciUuVrsk2kyhedokMd1KPKGJQREZHbKlcLSE7PxrqUq0hOz0a52nlJ2uTaPJQKPNE60mSbJ1pHwkPpvJoYnk47MxERkRWSUjMx49c0ZOb9s1dhpMoX0x6PQZ9Y0x++VP2UqwWsP55pss3645l4q08LpwVmHCkjIiK3k5SaiZd/PKoTkAFAVl4RXv7xKJJSTX/4UvVzMCOn0vNFX2ZeEQ6yThkREZE05WoBM35NM1hNSrxtxq9pnMokHTfumA7I5LazBwZlRETkVsyNeAhw/ogHuZ7wQNNJ/nLb2QODMiIicivuMOJBrkdcfWksW0wBrr4kIiKSxR1GPMj1eCgVmPZ4DIDKe46Lv097PMapqy8ZlBERkVtxhxEPck19YiOxaGg7ROjVK4tQ+WLR0HZOX7XLkhhERORWxBGPl388CgWgk/DvKiMe5Lr6xEbi4ZgIHMzIwY07RQgPrAjgXeH5ohAEgctTXFB+fj5UKhXy8vIQFBTk7O4QEbkc1ikjV2TN5zdHyoiIyC258ogHkSUYlBERkdvyUCoQ17ims7tBZBNuleh/9epVDB06FDVr1oSfnx9atmyJw4cPa+4XBAFTp05FZGQk/Pz8EB8fj/Pnz+scIycnB0OGDEFQUBCCg4MxcuRIFBQU6LQ5ceIEHnzwQfj6+qJ+/fqYPXt2pb6sXr0azZs3h6+vL1q2bIlNmzbp3C+lL0REROR4rrpnqtsEZbdv30bXrl3h5eWF3377DWlpafjkk08QEhKiaTN79mx89tlnWLx4MQ4cOICAgAAkJCSgqOiffIMhQ4bg1KlT2Lx5MzZs2IBdu3Zh9OjRmvvz8/PxyCOPoGHDhjhy5AjmzJmD6dOn48svv9S02bdvHwYPHoyRI0fi2LFj6NevH/r164fU1FRZfSEiIiLHSkrNRLePt2HwV/sxfkUKBn+1H90+3uYSW3O5TaL/5MmTsXfvXuzevdvg/YIgoE6dOnj99dfxxhtvAADy8vJQu3ZtfPfddxg0aBBOnz6NmJgYHDp0CB06dAAAJCUloW/fvvjrr79Qp04dLFq0CG+//TaysrLg7e2tOfcvv/yCM2fOAAAGDhyIwsJCbNiwQXP+Ll26oE2bNli8eLGkvpjDRH8iIiLbEvdM1Q98xCxEW5TFsObz221GytavX48OHTrgmWeeQXh4ONq2bYuvvvpKc39GRgaysrIQHx+vuU2lUqFz585ITk4GACQnJyM4OFgTkAFAfHw8lEolDhw4oGnTvXt3TUAGAAkJCTh79ixu376taaN9HrGNeB4pfdFXXFyM/Px8nR8iIiKyDXN7pgpw/p6pbhOU/fnnn1i0aBGaNm2K33//HS+//DJeffVVfP/99wCArKwsAEDt2rV1Hle7dm3NfVlZWQgPD9e539PTE6GhoTptDB1D+xzG2mjfb64v+mbNmgWVSqX5qV+/vrlLQkRERBKZ2zMVcP6eqW4TlKnVarRr1w4ffvgh2rZti9GjR2PUqFFYvHixs7tmE1OmTEFeXp7m58qVK87uEhERUZWRlS8tp1tqO3twm6AsMjISMTExOre1aNECly9fBgBEREQAAK5fv67T5vr165r7IiIicOPGDZ37y8rKkJOTo9PG0DG0z2Gsjfb95vqiz8fHB0FBQTo/REREZBs5BcU2bWcPbhOUde3aFWfPntW57dy5c2jYsCEAIDo6GhEREdi6davm/vz8fBw4cABxcXEAgLi4OOTm5uLIkSOaNtu2bYNarUbnzp01bXbt2oXS0lJNm82bN6NZs2aalZ5xcXE65xHbiOeR0hciIiJynNAAb/ONZLSzB7cJyiZMmID9+/fjww8/xIULF7Bs2TJ8+eWXGDt2LABAoVDgtddew/vvv4/169fj5MmTGDZsGOrUqYN+/foBqBhZ69OnD0aNGoWDBw9i7969GDduHAYNGoQ6deoAAJ577jl4e3tj5MiROHXqFFauXIn58+dj4sSJmr6MHz8eSUlJ+OSTT3DmzBlMnz4dhw8fxrhx4yT3hYiIiBwnPNDXfCMZ7ezBbSr6d+zYEWvXrsWUKVPw3nvvITo6GvPmzcOQIUM0bd566y0UFhZi9OjRyM3NRbdu3ZCUlARf338u8NKlSzFu3Dj07t0bSqUSAwYMwGeffaa5X6VS4Y8//sDYsWPRvn17hIWFYerUqTq1zB544AEsW7YM77zzDv7973+jadOm+OWXXxAbGyurL0REROQgUnffcuIuXW5Tp6y6YZ0yIiIi21l77ComrEwx227uwDZ4qm1di89TLeqUEREREVmKif5ERERELoCJ/kREREQuIELlZ9N29sCgjIiIiKq8TtGhiFSZXmwXqfJFp+hQB/WoMgZlREREVOV5KBWY9niMyTbTHo+Bh9J5yy8ZlBERERG5AAZlREREVOWVqwVMXnPSZJvJa06iXO28SmEMyoiIiKjK25+ejdy7pSbb5N4txf70bAf1qDIGZURERFTlJf95y6bt7IFBGREREVV5UvcvcuY+RwzKiIiIqMoL9pdWFFZqO3tgUEZERERVXlgNacGW1Hb2wKCMiIiIqjxW9CciIiJyAazoT0REROQCPJQKPNE60mSbJ1pHsqI/ERERkT2VqwWsP55pss3645ksHktERERkTwczcpCZV2SyTWZeEQ5m5DioR5UxKCMiIqIq78Yd0wGZ3Hb2wKCMiIiIqrywAB+btrMHBmVERERU9UnN33denj88nXdqIiLHKVcLOJiRgxt3ihAeWLHs3ZmrrIjIsW4VFNu0nT0wKCOiKi8pNRMzfk3TSfKNVPli2uMx6BNreok8EVUN4YGma5TJbWcPnL4koiotKTUTL/94tNKqq6y8Irz841EkpZpeIk9EVUP7hiEwNziuVFS0cxYGZURUZZWrBcz4NQ2Gqg6Jt834Nc2pdYmIyDGOXLoNcy91tVDRzlkYlBFRlWWuLpEA59clIiLHYEkMIiIncoc3YSJyDOaUERE5kTu8CRORYzCnjIjIiTpFhyJS5Wu07JACFaswO0WHOrJbVM2VqwUkp2djXcpVJKdnM6fRQdwhp4wlMYioyvJQKjDt8Ri8/ONRKACdhH8xUJv2eAzrlZHDsDyL81y7fVdGu5r27YwRHCkjoiqtT2wkFg1thwiV7hRlhMoXi4a24wchOQzLszhXyl+5Nm1nDxwpI6Iqr09sJB6OiWBFf3Iac+VZFKgoz/JwTASfl3bj+vssMSgjomrBQ6lAXGPnTEkQySnPwuepfdQP8bNpO3vg9CUREZGdsTyL85UL0hZUSG1nDxwpIyIit+UuG82zPIvzHZW4qlJqO3tgUEZE5KbcJSCxF3daySiWZ8nKKzKYV6ZAxeITlmexn7sl5TZtZw8MyoiI3JA7BST2IK5k1A9wxJWMrrayluVZnE+tlhZsSW1nD8wpIyJyM9W9tIK7bjTP8izOVSwx1pLazh44UkZE5EZYWsG9VzKyPIvzNAj1x9HLuZLaOQuDMiIiN+LOAYmtuPtKRpZncY6n2tbFLynXJLVzFk5fEhG5EXcPSGyBKxnJIlJns504682gjIjIjTAg4UbzZJm1KVdt2s4eGJQREbkRBiT/rGQEKm+Iw5WMZMyVHGkbkkttZw8MyoiI3AgDkgpcyUhy+XpJC3mktrMHJvoTEbkZMSDRr1MWUY3qlAFcyUjyhNWQNqUvtZ09MCgjInJDDEgqcCUjSRWh8rFpO3tgUEZE5KYYkBBJV3CvzKbt7IE5ZURERFTlZeYX27SdPXCkjIiIiOyiXC24zBT7n7cKbNrOHhiUERERkc0lpWZWWowS6cTFKP4SV1VKbWcPnL4kIiIim0pKzcTLPx6ttCVYVl4RXv7xKJJSMx3epzrBfjZtZw8MyoiIiMhmytUCZvyaZnC3IvG2Gb+moVzt2P2MpI7OObOkDIMyIiIispmDGTmVRsi0CQAy84pwMCMHQEUQl5yejXUpV5Gcnm23YC1SJW0ETGo7e2BOGREREdnMjTvGAzL9dg7NO5O6vsCJpf44UkZEREQ2Ex4orSL+xVt3HZp3dt3E6J0l7eyBQRkRERFZRXsKUi0IiAjyMTrgpAAQEeSD5QcvOzTv7NiV2zZtZw+cviQiIqqGbFVDzNAUZLC/FwRUBGDaYZV49MGdGmDulvNGj6mdd2arXSvKBWkBntR29sCgjIiIyM3JDbBslcsllr7QD2Py7pYCAFT+Xsj9+/8BIOLvcxSXqSUdX2p+mhQeCmkBp9R29sCgjIiIyI3JCbDK1QIWbDtvcJRKzOVaNLSdpMDMXOkLBQA/Lw8sHNkOtwqLdYLF5PRsSX+b1Pw0Q33TD1Jb1wvGD7hs9rGt6wVbdE5bYFBGRETkpoyNVBkKsJJSMzF9fRqy8g2PPomB1Ixf0/BwTITZqUyppS+USgWebFNX575O0aGIVPkiK6/IYFCnQMWoWqfoUJN9MMRYkBon8Vg5BSWyz2krTPQnIiJyQ3KKtIrBm7GATPtx2jXETJFT+kKfh1KBaY/HAKhcgUL8XbxfTg0zUzsJrEm5Jqm/p67lSWpnDxwpIyIickNSR6r2/5ltNHgzRkrAJXVq0Vi7PrGRWDS0XaVRLZW/F0Y8EA21Guj28TbJeW9SglQp/sq9K6O1bXGkjIiIyA1JHalKTs82GbwZIiXgEqcgTZW+iDQzBdknNhJ7JvXChPimCPbzAgDk3i3F3C3n8H/L5NUwMxekSuXj6WH1MSzFoIyIqBpx1JY2ZH/Sk+Cl/xtLCaREUqcgzeWmbU7Lwrwt55F7r9RkO8B0DTNbrdSsGeBjk+NYwm2Dso8++ggKhQKvvfaa5raioiKMHTsWNWvWRI0aNTBgwABcv35d53GXL19GYmIi/P39ER4ejjfffBNlZWU6bXbs2IF27drBx8cHTZo0wXfffVfp/AsXLkRUVBR8fX3RuXNnHDx4UOd+KX0hInKkpNRMdPt4GwZ/tR/jV6Rg8Ff70e3jbTavnE6m2SowljpSFdcoTNZxpQRSInEKMkKlGyBGqHwlreI0NeVojLG8N0tXaupTWlCrzVbcMqfs0KFD+OKLL9CqVSud2ydMmICNGzdi9erVUKlUGDduHPr374+9e/cCAMrLy5GYmIiIiAjs27cPmZmZGDZsGLy8vPDhhx8CADIyMpCYmIgxY8Zg6dKl2Lp1K/71r38hMjISCQkJAICVK1di4sSJWLx4MTp37ox58+YhISEBZ8+eRXh4uKS+EBE5kpxVemQ/ttzrURypevnHo0aLtE57PAZdGtc0udJRFBHkg+lP3C+7H31iI/FwTAQOZuTgWu49pPxdEf/q7XsoKVPD21N3/Ee7XMWtO8UWTznqj4yZW9HpDhSC4MTStRYoKChAu3bt8Pnnn+P9999HmzZtMG/ePOTl5aFWrVpYtmwZnn76aQDAmTNn0KJFCyQnJ6NLly747bff8Nhjj+HatWuoXbs2AGDx4sWYNGkSbt68CW9vb0yaNAkbN25Eamqq5pyDBg1Cbm4ukpKSAACdO3dGx44dsWDBAgCAWq1G/fr18corr2Dy5MmS+mJOfn4+VCoV8vLyEBQUZNNrSETVS7laqJQwrU0sP7BnUi+LKrqTNMYCY/GKywmMtQObi7fuYvnByzorK/UDPfHcgOHJzBo+npg9oBX6trI8MJ+1KQ1f7c6A9sCfUgGMejAaU/rGaPqhH5RaavmoLpWq/Rv7O/WDVlPG9IjG5EdjLO6XNZ/fbjd9OXbsWCQmJiI+Pl7n9iNHjqC0tFTn9ubNm6NBgwZITk4GACQnJ6Nly5aagAwAEhISkJ+fj1OnTmna6B87ISFBc4ySkhIcOXJEp41SqUR8fLymjZS+EBE5itRVelLKIFiruua0lasFTF8vrXyFOfrT0HO3nENxWTkeja2NcQ81wdJ/dcaeSb10AjxxmlHl72XwmIXFZRi7zPJNwGdtSsMXu3QDMgBQC8AXuzIwa1Oa0XIVcpnKezM1nTqgXd1K7Q2pGWCbaVBLuNX05YoVK3D06FEcOnSo0n1ZWVnw9vZGcHCwzu21a9dGVlaWpo12QCbeL95nqk1+fj7u3buH27dvo7y83GCbM2fOSO6LvuLiYhQXF2t+z8/PN9iOiEgua+pJ2ZItp+4sZYv9Hi05xoJt503WCJO616Ox0bbbd0vxW+p1ANfx89G/Kl3TcrWAQB8vGJsbk1s4VltJmRpf7c4w2ebLXRmoHZRp9bSilAUE2tOp2v9GPx+5gp+PXjV7DpWf80IjtwnKrly5gvHjx2Pz5s3w9XVeFGsvs2bNwowZM5zdDSKqgqytJ2ULUnLaDH2Q2nI6Ve52RIb6YklgmZSaaXLzbW3GAuNytYD9f2Zj8s8nzQY2mXp5glKnDDV1zdKz0bWp9MUBPyRfrDRCZujY5grXShEhMYj3UCoqBbeb06Qtttucdh3PdmxgcR+t4TZB2ZEjR3Djxg20a9dOc1t5eTl27dqFBQsW4Pfff0dJSQlyc3N1RqiuX7+OiIgIAEBERESlVZLiikjtNvqrJK9fv46goCD4+fnBw8MDHh4eBttoH8NcX/RNmTIFEydO1Pyen5+P+vXrS7k0REQm2XNLGymk7JE4Zc1JTF9/Cln5/8wY2HIUTe52RIYCrydaR+LLXRmyFkuIf7tUhgJjS/OwZvyaBrUaGLus8t9tythlR/HRgJaSr/ulHPsUW41U+eLdxBYICfCxSaBeWFJmvpGMdvZgdVCWn5+Pbdu2oVmzZmjRooUt+mRQ7969cfLkSZ3bRowYgebNm2PSpEmoX78+vLy8sHXrVgwYMAAAcPbsWVy+fBlxcXEAgLi4OHzwwQe4ceOGZpXk5s2bERQUhJiYGE2bTZs26Zxn8+bNmmN4e3ujffv22Lp1K/r16wegItF/69atGDduHACgffv2Zvuiz8fHBz4+zquNQkRVl9RVevZK8peS03b7buUaVbZaGSolKHx7bSrulZTjcs49zNtyrlLbzLwifLHL8BSdqak/OQVNDeVJGQsmzRFHvd5Zlyr7sbn3SvHyj0ex8Ll2CAnwNhsQNQz1l3kG495NbIGwQB+7jJQGeEsLeaS2swfZZ3722WfRvXt3jBs3Dvfu3UOHDh1w8eJFCIKAFStWaIIQWwsMDERsbKzObQEBAahZs6bm9pEjR2LixIkIDQ1FUFAQXnnlFcTFxWlWOz7yyCOIiYnB888/j9mzZyMrKwvvvPMOxo4dqwmIxowZgwULFuCtt97Ciy++iG3btmHVqlXYuHGj5rwTJ07E8OHD0aFDB3Tq1Anz5s1DYWEhRowYAQBQqVRm+0JE5EjGtrSROh1kDUtz1azJc9ImJSjMLizBhFXHLTq+eAxDOWFy/nb9wNiSGl76cgot21xbADBu+VGdaUljI5fPx0Xhg02nTU5hKgDUDvLF9XzTo7UvdI2225eDh5qFY/PpG5LaOYvsoGzXrl14++23AQBr166FIAjIzc3F999/j/fff99uQZkUc+fOhVKpxIABA1BcXIyEhAR8/vnnmvs9PDywYcMGvPzyy4iLi0NAQACGDx+O9957T9MmOjoaGzduxIQJEzB//nzUq1cPX3/9taZGGQAMHDgQN2/exNSpU5GVlYU2bdogKSlJJ/nfXF+IiBzNWAK0vctgWJOrZizYkZNsvznN8AIre7hxp6hSHS4pnm5XFw/H6Ka32GrbIEvpB1nGRi69PZUY9WC00ZFEABjdPRptG4Q4bbQWALadkZZTtu3MdTzXpaHd+mGK7Dplfn5+OHfuHOrXr49hw4ahTp06+Oijj3D58mXExMSgoKDAXn2tVlinjMi2bLHqrjqz5vqJddKsKeo5f1AbPNmmoqSBnGT7pNRMjPm7bpUjTIi/DysOXdbpm1JROcAxRP9vWJdyFeNXpFjUDwWAkAAv5BSa37pI7nGN1bSztE6Zo1bg9vx4Gy7evme2XVSIH3ZM6mXxeaz5/JY9Ula/fn0kJycjNDQUSUlJWLFiBQDg9u3bVXJVJBG5P1coxWBLjg4wrb1+pnLapBJH2+Qk7MtNsreGAoDK38tgPprUUmz6f4OlI4ziM+H9J2Mxc+Npm1a4N1W6Y0rfGLz+SHP8kHwRl3LuomGoP56Pi9Kp6O+s0VoAKCiVlsAvtZ09yA7KXnvtNQwZMgQ1atRAgwYN0LNnTwAV05otW7a0df+IiKxS1bYXcnSAaavrJ+a0TV5zErl6Sf2mAjXtlaFSEva1888cNf2n3X9TwY+5ETP9v6FTdCgignxll5LQzhNUKhVWBcPGGMqVK1cLOHLpNsICfRBTR2U02DJUrsIR6ql8cavA/MhhPZXzBphkB2X/93//h06dOuHKlSt4+OGHoVRWRMCNGjXC+++/b/MOEhFZSu6HuLFj7E/PRvKftwBUfJh0aVTTKVOfxgKkzLwijPnxKBbbOMC0xfXTpx+QiccyRD/XKDk9W/LOBHGNa2KLnXLJ9IOrCJUvBnWsb7YWmVoAnu/SAD/sv2y0jaZW2J/Z6NokDIM7NcDcLefM9snYqkVjCzy0S3yI55Xj1p1ilKsFzXkMfVkI9vPCiK7RGNeriUukCsTUDUbK1TuS2jmLRes+O3TogFatWiEjIwONGzeGp6cnEhMTbd03IiKryNleyNA396TUzEojOwu2X0Cwvxc+6i+9jpMtlKsFTF5junDo5DUnrVqlqM/a66dNylSioWBHewRQzs4E5WoBa1PMV2+3hJiJ/WLXKM2I1oYT1yQ9VqGQ9m8zdmlFrbCoMGnlJm7fLdVZuag/xb3zzYdw5NJtg1OGX+3OMFrp35iZG0/j6z0ZmPZ4Ra6YoS8LufdKMXfLOSzZl+Hw14sh7RqEYNnBK5LaOYvsoOzu3bt45ZVX8P333wMAzp07h0aNGuGVV15B3bp1MXnyZJt3kojIEtZsL2QqQTz3bqldRqZMWbDtvMFRJv1+Ldh2AePjm9rknFJXLRqbytIOCtSCYHYqUS2YrlMlZ2eCgxk5Nk9yF4mjhL+lZuHtxIpRPKl9k1rTS6wV9prEf8sF2y9otlcCYHSKW1wsAVQ8xw0Vw5Uq6+8R2mB/L5PHcMbrxZDbd6WVB5Hazh5kB2VTpkzB8ePHsWPHDvTp00dze3x8PKZPn86gjIhchqXbC1VsHn3K7OOsrZ8lVblawJK9FyW1XbIvwybTRUmpmfhW4jn1r5+xqSwpwgJ9dAIHbXJ2Jvhwo30T/LWnGZUKBbLyixBqYrWj2Lfn46Lw1e4MSXliAoDlBy8jwkR9L21ikGTqvgnxTREVFoCwGj6Yvv6UVXlm4mPNfVkQOer1YozUIN1ewbwUsoOyX375BStXrkSXLl10hmHvv/9+pKen27RzRETWsHR7oYMZOTrb/RgjderOWgczcpB7T9oHRe7dUqv7JGfVon4VemN5b1L7by6QHtTRcI6Vdv7Z5rQsfCMxoLTW6B8Oo7C43GQb7b55eyrRISoEG05kSjp+Vn4xJsTfh3lbzplN1pdyn9Q9OO3BUa8X4+c3Xw5DTjt7UJpvouvmzZuaLYq0FRYWSp4rJyJyBLEUA/DPB6PIVMFKOVXYLa1WL4fcc1jbJzmrFrWvnzUV6BUwvM2QKCk1E90+3mY06T3Y30uzqbklZTBCA7zQJ7a2+YZ6zAVkAFDD1wM97gvD1dv3cK+kHHvO35J1jqgwfywa2g4RTlwVaCuOeL0YI7Usq8zyrTYlOyjr0KGDzpZDYiD29ddfG93XkYjIWcTVZ/ofaBEqX6PlHOTUh7KmWr0p5WoByenZWJdyVXJVeFv1SeoH58iuUTrXz5oSFAKAJ1pHGpzaEkffTB1b3DvT0j7kFJbKDpbMiarpB4UCuFNUjh3nbmHmxtOImZYkedRQFB7oiz6xkdgzqRfGPdTYpn10NHu9XiSdO0ja/tJS29mD7OnLDz/8EI8++ijS0tJQVlaG+fPnIy0tDfv27cPOnTvt0UciIqvILVhZUR/Kx+wUpqmRHWsYysmSUhXe2HSsXFI/OOP1tgWydhTky10ZaNsgRCfQkzr6JpbneKtPc4vPXyBh1EuOi9mVp8HkDsJoP8c8lAp0bVILC7a7X6qQrZ6b1si4WWjTdvYge6SsW7duSElJQVlZGVq2bIk//vgD4eHhSE5ORvv27e3RRyIiq4kFK59sUxdxjU3XGfNQKjD9ifvNHtPcXn3ao13J6dkol1Da3diokNSq8LbYP/B2YTFMHcLYVKMtRkFm/Jqmc52kjnyJifc5BfJGFV2d/uihWFDWkR5vFQEFKqcASOWovS3NcYecMovqlDVu3BhfffWVrftCROQy+sRGYrGRCvQh/l6YZabukqHRroggXwzu1ABRYf4GR+ukjAoZGzGzVVX/pNRMjF12zOzIlKEP2E7RoQj285I9PScyVPcsS+YHZGiAt8nFHe5m5aG/0P2+cE3B4s1pWSgqs+2InilKBfBobB0ktqpT6fkslX69OWcpLFHbtJ09yA7KLl82XokYABo0aGBxZ4ioenCHzcHL1QJUft6Y9vj9uHWnGLfvFkOpUEqq6G90a6L8Ip1Edf1ASsqokFjLKzTAGzmFJQit4YOIIPPXUMo1lxoULhhsOBfPQ6nAiK5RVq/w23vhJm7cKcLFW3fxffJFWY+NUPlZvc+mK8m9V4ohXx/QqcDvyL9JLQBjl1Vsp7VnUi8czMjB4p3p2HnuptnHDotriEdjI53++haf++pyacFssK9F41U2IfvMUVFRJldZlkv8o4mo6jMUCGxOy3L5zcFN7S/ZtUmYycfKWYGov3+k1JwsU7W8DDG3X6b477T3wi1JQWFIgLfR+8f1aoqvdmegoNjyTZ0tyZnSzlnyUCoMbi3kzjLzivDF31siWcrSIFWA7n6cY5cZroWm79HYSJ3yF9rvB2E1fAABuFVYbPEXMylfNAw9980pc2IkLzsoO3bsmM7vpaWlOHbsGD799FN88MEHNusYEbk3gwVE/b0MFpo0t7m1+OablV+RMxQa4I0IlZ9dvoFL2YDb1KKB/Wb2Z9Smv3+kpcVurfl7RnePxvrjmbI+tEwFjx5KBZ7tUE9y4VlbEqdUxVHOtxKaaUYT028UYMH2Cw7vk7OJrw5L/p21ZeYVYcG28+gUXRM5heYr3ocGeFWqXzd9fZrRorlyv5iZ+6IhtjH03DendqAbrb5s3bp1pds6dOiAOnXqYM6cOejfv79NOkZE7stoAVEjlb9NbW5t6puurUfYpGzAPWXNSUxff0pnZabYDwCY/PNJWefUzqMyV+wWqPiwa99Q2t585v4eABaNvly8ddfkOesG+8k+prVGd49Gn9hIox/WgzrWd3ifpAj280LevVKbTUnq5/QF+npiQPt66NmsNl5/pDmOXLqNxTsvYOc5+eU/5m45j5FdpeULPtWmrs5m5cZ2GhCZ+2Kmzdj7S6beFydLa+ZZvKLBBmSvvjSmWbNmOHTokK0OR0RuytICotrBichcfSrxTTgpVVp1dHOkbMB9+25ppVIZ4hY2Y348anGS+407RSaL3YpyCkvRffZ2SX+zNXXDTJm35ZzB84sFXmduPG3zc5qz/ngmNp24ZvD5kpVXhLlbzsPH02YfeTYT3yLcpjliwx+IQmjAP1ta5ReVYcneixj81X70mLMdtwtLcPJqvsXHl7rJu1gupVwtYPIa819UxGugv/pWn7n3F3Gqdf+f0kesKx1D7bxEf9nP0Pz8fJ2fvLw8nDlzBu+88w6aNrXNJrhE5L6sDQTEqTGpwZ34Jiyl3ITUc8tliw9VcUrSWLFbbVn5FUGgucDMntXT9a+5lAKv9pSZV4R31qWaHBUsLnPeh60hSgXw01FpQY5U87eeN7p3Y2ZeEf5v2VFJ04/G5BSWIjTA2+iXBv1yKfv/zJa8N6ahL2b6pLy/ZOYVITk9W9I5DckwUF/OUWQHZcHBwQgJCdH8hIaGIiYmBsnJyVi0aJE9+khEbsTaQEAMTuQEd+beyOWe25EM1fzqExuJnW8+hBo+pjNMxq9IwVe70lFiJNiw19+j/+FpzfZKtuTMjaQtYYPvEU7xZJs6AKRtXWZJcGTqPURqiRRrtkry9XLe/KXsnLLt27fr/K5UKlGrVi00adIEnp7OW0ZKRK7B0kBAv+K3o/d7BMxvYG5rpopqHvgz2+wKxuIyNT7YdAazfjuDUQ9GY0rfGJ372zcM0ZTOsAfxmttrmrSqUijkV/Z3JetSrhpcOGC4Hpn8P9TUe4jU53Kwv5fFr2U/Lw+Zj7Ad2VFUjx497NEPIqoiLAlsDAUncoM7W4wKiTld9qpxFeCt1ClMaayoZlJqJl5ffVzycdXCPwn7YmAmJrzbKyAD/rnmztxk2h25c0AGVIxIfrkrAwufa4uQAB9k5d3TrHJV+XlrprUPZuTI+lulbMUUWkPaysiwQF/Na1mukADn7c8pKShbv3695AM+8cQTFneGiNyfqcBG/F2/NIah4EROcGfLPSjFnC5DK/julZYj767lK+UKS9R4NLY2GtcKNFiEtlwtYMG28xYXX/1qdwZef6Q5tp25brYUQA0fT4triYkfnu0bhiA5PRvnr9+x6DiuLMjXA0ql0qp/76pu5sbTeDcxBrN/P1up9A1gfLW1IVK3YpK6xVREkC/iGtfE6O7RslcY+3s5b0GIQpAw8apUSuugQqFg8Vgbyc/Ph0qlQl5eHoKCgpzdHSLZTNURkro5uJQ6QwpA0jJ6uYwVvhW/eRsKNuXQvxab07LwS8o1q0e2+rWpg13nb9o9v6p381o4diXX7fK4pHq7bwvUD/Wz2b83mSa1vE25WkC3j7eZnC6PVPliz6ReAGC2rSFt6gfhl7EPynqMNms+vyUFZeR4DMqoKrDFdkqOrFNmiP7fcLuwGDM3njYYbJ7NKtDZRskUY6OG5BrmPtsaT7WrZ/TLhbjlEcAAzVIh/p7o37Ye4mMi0L5hCI5cui3pvcJY3TOxtfglLTk9G4O/2i+7X/WC/bBnci/ZjxNZ8/nNzHwishsPpUJnmxVL9ImN1Iwm2aOiv6nAcdOJTLyzLlVn9CpS5Yu3H22B63eKcCnnLhqG+uP5uCh4KBVYlyI9f0X8IGdA5poiVBUFcLWff/rPkbYNQioFbLUDvVFQUo7CYtebNTK2mb2jjXuoMbo2qaW5jkmpmegxZ7usrdcMfZlR+Xvho/4tNY+xNNfRU+m8i2RRUFZYWIidO3fi8uXLKCnRHWp/9dVXbdIxInItztxE3BbBnSGmpliPXb5tMBclM68I41bobjf33+0XAEFA7j3L93sk16Gfo2jo+afZyqlPc80Xhcs597D84GUUFls2BR3g7YHCEtsEc0+3q4uuTcJ0Nq2/XViMscsqnrvOjM2a1g7UXE8p25ppB2amUhry9II0Sxf/1A5yo22Wjh07hr59++Lu3bsoLCxEaGgobt26BX9/f4SHhzMoI3IwRwRLUvaZs5Qjgz3tc128VWgwoV6szi8HR7uqDgXMJ5vL2ddVCu2Ro/9uPY95Wy1b6CEK8ffCx0+3Nvg3LFIqDI7u3SgokbxS0tqcOjFYkrKtmfbWa1Lq4Wm3t7TETYkTv1vJDsomTJiAxx9/HIsXL4ZKpcL+/fvh5eWFoUOHYvz48fboI5FLcubIkciewZL2OeR8k5V7bHv339S5DHGB2R1yEu3nnrHXt9x9XU0RV7FOeLiZ5r3jld5NseLQ5UpbeWk/RuXvpRkV0u+HAsCs/i2NvhcZmo5VCwKGfH1Acr8jVL54N7FFpdxKc/RLXkjZ1kwsUhzXuKbk9t/tzUBYoA/CAyv6OXbZMVmB5O179isjY47soCwlJQVffPEFlEolPDw8UFxcjEaNGmH27NkYPnw4NySnasGRwYSpPpgLlqSucjRG7jdZ8THWrKzM/Huk6vPn2qJvqzom+6Z9HlOJwlJWcVL19WhsbQyLi9YJvCqNhPl54YUHorDi0GWbPI+MlYDwUCow/Yn7ja76BICP+rcEAIvfg/SnY9dJ3M9yWFxDPBobqblOx//KlVxuwtDfKzXnS2wntb323quRKl+DhW5NcuIbheygzMvLS1MiIzw8HJcvX0aLFi2gUqlw5coVm3eQyNXYc+RIKinB0uQ1JzF9fRqy8i0PHOV+k5UarEqZhhi7/BjGXy/AK72bVgrqDJ1HP4lZu+SEK2wBRK4p1N8LC55rr5kem7/lnMFp7dx7pVZPK2ozVjhYzFUb0TWqUokU/cdY+6VLlHGzUFK7R2MjNcFcuVrA+uOm917VZujvlZrzJbazJEcsK6/o70K37bBw+3mcyjRfU69uiJ/s89iK7KCsbdu2OHToEJo2bYoePXpg6tSpuHXrFn744QfExsbao49ELsOSkSNrz2foTVdKsFQxnaI7pSI3cJTzTVZOsCplWx5BAOZtPY/vki/qrKgydh79VWXieV+Lb8otgMio9/u11IyO6X+JsZd+bergmfb10UVv8YChLxuhAV54qk1dxMdEVAq6bLEAZtOJa/hsm/lgM9TfC1l595Ccno1O0aGSt9bSX2mpzVzOl/50pyU5YuL78syNafjwyViM+N9hs48Z/WBjiUe3Pclla8WisB9++CEiIyveHD/44AOEhITg5Zdfxs2bN/Hll1/ap5dELkLOyJG1klIz0e3jbRj81X6MX5GCwV/tR7ePtyEpNdPipd7iG9mMX9M0W6GYIvWbaVgNH5PBqv455fQ/924pxvx4FEmpmbI2vhb+/vli55+Sz0XVy0vdo9G3VaQm0HdEQAYAv6Rcw5BvDqDjB1uw6UTFaJPYB/33l9uFpfh270Xk3SuxywKe/1t2TFKZjJy7pZiw6rjmfWhLWpakc4grLQ31Xdz9A5C2ubmp9qaI78uentJCnrgmYTKObluSg7K6deti8uTJCAoKwkMPPQSgYvoyKSkJ+fn5OHLkCFq3bm23jhK5Ark5EJYy9gYtjv5cvCVtusEQOYGj+M3U2BugAhXThBAgK1i1ZBpixq9p2J+eLXvU626p69WLIufz9VSieaQKe8/fwvT1zpnezikswf8tO4oPNqbJ+lKjr1wtIDk9G+tSriI5PVvSFy7xC44lsvKK8M3ei5LamnutPxwTgdfi74PKz0vn9giVr8ERfXEbtAiV/PeQAxnZktoduXRb9rFtRfL05dixY/H9999jzpw5eOCBBzBy5Eg8++yz8Pf3t2f/iFyK3BwIS5SrBUxff8rkFOnyg5cREeSL6/nylnprMxU4ak+bDurYAPO2nDO4jyVQ8U32VqHhlWLGzikGe3ICrMy8IiT/eUtyeyJTisrUmLAyxWbHs2aHhq92m06W18/d1GbpoiOp04/G+gNU5HEKguG8eCmbixtbUDGiaxTG9aqcSyrSX0G6+9wt/HT0L7P9LpdY8+Na7j1J7exB8kjZu+++iwsXLmDr1q1o1KgRxo0bh8jISIwaNQoHDkhfSkvkzqSOHFmzOfaCbReMLocHKt4As/KLMbhTA805LaEdOGp/056/5Ry6frRVM206d8s5qPy9oPI3/E324ZgI7D53U9I5wwIqijJ6KBV4N7GF7D6fq4IbX5M8ji06I12EyheLh7bDwsHt7HYO/S9S5kbUk1KNJ+JbO5oPVORxil8UtUnZXNxY3/PulWLelvPYbGZ6VMyn8/FU4mczAZn4vpwvcZ/WY5edN1Imeyv0nj174vvvv0dWVhY++eQTnD59GnFxcbj//vvx6aef2qOPRC5Dbg6EXEmpmZL3TowK8zc4jB8R5INgfy+zgWP7hiFITs/Ge7+eQscPtmgFYecrBYV5d0uRe7cUT7eri2FxDfFuYgvsfLMijaH9+5vx01FpS+pfX31c80EREiC/avYfaTdkP4aqFldaRRvq74W5z7bG8lFdsGdSL/SJjZQ8amyJW3eKNVOUJWVqq6Y8rRnN1/Zi16jK70FGph5F5hZMAdLyXqXmmAoA3k1sgTMSv9Rdd1BuoSE22ZB848aNGDZsGHJzczULAsg63JDctdmqTpn2NGFYgA9eX31ccrLx8lFdENe4psEVmpvTskzWORrdPRrrUq6ZHJEzx5JpGu0Ng4vL1Bi/IsXi85PrCPbzrLZbTImvQ5Glm2DLFejriTtF5q/5u4ktNIVUtVdAlqsFdPt4m9Urk5eP6qJZjXnjThHCavgAAnCrsNhomQ6p10j/2uqTepzHW0Xg8KVcyX9rvzZ1MG9QW0ltDXHKhuR3797FqlWrsGTJEuzZsweNGzfGm2++aenhiNyKqU2KpZJaYd4Q7SlSQ8vixWRY/eNHqHzxROtIyQUfTbGkgrl22ZD/PM2FQVXBY60iEanyNZsXVVXpTwNaurWPXFICMqByIVXxi6OHUmH1e4H4PiS+B206kYlXlh/Tqa1m6MuqrRZMST3OryekrRQVPWmiaLW9yQ7K9u3bh2+//RarV69GWVkZnn76acycORPdu3e3R/+IXJaxGkFSKtpbW2FeyhSpocCxZV0VOnyw2cKz2oaYtHzoorSVUOTadp+/ibxqOkoG6O7jKL7WBnaob9NCs7Yi7paxeGhF3tuXVgRk+nuEztqUZjDAyzRQp9BWC6ZsNQWr79zNAjyE2nY5tjmSg7LZs2djyZIlOHfuHDp06IA5c+Zg8ODBCAwMtGf/iJxK7v6WUqY15dTaMmRCfFPJU6TageOsTWl47uv9kjcdtrcvd7F+WFVQnQOyYH8vqNUCNp3IxMyNlo16O8Okn0/Az8vT4vegEH8vzNIq6LzpxDWTI24C5G0ULmXlJmC/UclDGdl4qYdzCshKDsrmzJmDoUOHYvXq1azcT9WC3LwxqRXtrVmKHqnyxbheTWU/zti3WGe6W6p2dheIrJJ7txRDvnG/6gN598okBdP6ZXAMlasoVwt4Z12q2WNpl/QQF0y9/ONRk6V2zM0GmDuOpYGaNbm21pIclF27dg1eXl7mGxK5GLmjXYD8/S3lbL9kyVJ0a1Z2lpSpq22+DxFZTly1aGihgOhgRg5yJJaa0H7vM5X3KmfBlKnjDOrYQPJqdm21g+SvDLcVyUEZAzJyR5askrRkf0s52y9Zkgch941K2w/JFyVto0JEhql8PdG7RTjWHrvmUiU59AX7e2Fsz8b4YNMZmx0zLNAHT7apa/R+OV8y9d/7bLFgytRxAGDFoct2X3RhSxavviRydXJHu0RyAiwxX0vOaqLHWtUxm09RO8gHnzzbBrcKjC8rl2rXeWmFXYnIsLyiMqw5ds3Z3TAr924pYuqoEOznhdx78ldHG2JuSzepXzIDfT3RvmFIpdutWTBl7jjlagGDOtbH3C3yFl04s0Cx7OKxRO7AmuKElizXlrOayFwBWgHA4E4NbBKQlasFHL2ca9Fjicj93CooRremtttQe/nByyaLuIrJ9ubcKSpDpw+3mNxlQJSUmoluH2/TFLQWN0GX8lj9Y8gNyACghq/zZgYZlFGVJGe0S58ly7Xlbr9kbFNdlb8Xgv29MHfLeYvfjLS3TPpub4bkekZEZD8KQLPThj1HYsJq+ODwRdttE5SVX2zwfVIkfsmU8jfl3i3FGDPbP1mzdZS5Y0g1oF09ix5nC5KmL/Pz8yUfkNXnyRVYU5xQ7nJtcZj90dgIfLv3ouTVRPp5EBdv3cW8LeeMTre+Fn8fosL8TY6eJaVmYvr6U05dPUREusRX6kf9WwIAJq85Kbn4cqTKF/95ujVu3CnCzI2ndQqz6p8jQuULCJC8K4hU5t5PxS+Z09enSTq3fk6uyJJ8Xn3Wlhzy8VTigSa2G2mUS1JQFhwcDIVCWmzPbZbIFVhTnFDOcm1DCwkUCujUAjOVpC/mQYhbnpiabtVeRWRosUJSaibG/L21EhG5Du33gHK1gOnr0wCYD8rEAq1d/56O9PP2MLl92rTHY+yy96aU99M+sZEI9PGSVCJEPydXZEk+rz5rSg4BgL+3h8WPtQVJQdn27ds1/3/x4kVMnjwZL7zwAuLi4gAAycnJ+P777zFr1iz79JJIJmuLE0pZrm1sIYGYfjGyaxTiYyIk5YTJfSPRX6xQrhYwec1JyY8nIttRKoDPBrZFzUAfs/s/HszIkTSaFBrghQ+faqnzxUvK+1Jyuu12yjD3PqmfjC9nJaahtrbYfsmSkkPabt8tNRn02ZukoKxHjx6a/3/vvffw6aefYvDgwZrbnnjiCbRs2RJffvklhg8fbvteEslki+KEppZrSxki35SahX8nSqsr9vXudCl/lob+UP7+9GyL9qIkIustGNwWfSXulyg1aHj3sfsNjq6bKyNhqyr34vvmo7EV59L/cmloliA0wFvy8Q2Nvtli+yVbbL2UlXfP6mNYSnaif3JyMjp06FDp9g4dOuDgwYM26RS5Lu0k8uT0bJOrcpzNWDJ9hMrXaDkMfeL04pNt6moqUQPSRraMLSTQt+lEJraekV+2QhzK3/9nNlYfuSL78URknWA/Lywe2k5yQAZIDxoigoy3M/a+JN5nbHW3HGLG0rd7L1ZacGQskd5Yvpu+SCOjb+YWTAH/bG1l7LOnU3Qogv2tWz15q8ANKvqL6tevj6+++gqzZ8/Wuf3rr79G/fr1bdYxcj2WFGJ1NlsVJ9Qn9ZuUdjtDdXcASNqixJT/+/EI8rjCksjhXnggSvJ7n/j6z8q7h9AAb9wuLLFq30dTjE1zRqp88UTrSKw/nlnp9ncTYxAS4I0taVn4Zm/lgtNiysTC59pi5sbTVo3CGZulMDXDIRK3trLnZ4+tarxZQnZQNnfuXAwYMAC//fYbOnfuDAA4ePAgzp8/j59//tnmHSTXYGkhVldgrDihNaR+IxTbGRvq7xwdIvlYxjAgI3KOeVvPo3lkoNn3PkOvf0Os2U5Nn6kvpG/1aWE0LWPiqhSDxxNTJt5Zlyp5WyVDJsTfZ/J6GQso9Rn77DmYkePWqRyyg7K+ffvi3LlzWLRoEc6cqdjK4fHHH8eYMWM4UlZF2WKZclXz121pI2WhNXyMBrQ5hSX4LfW67TtHRA6h/95naDR8c1qWwde/IdZsp2aIsS+kxm6XsvrRmoAMAKLC/M22EQPK/X9mY+zSowZHrox99lib6A8AKicWj7Vom6X69evjww8/tHVfyEXZYpmyq7Jks/JytYBfUq5KOn5YgDfe+vmE2+y7RmQJX08lisrUzu6Gw2m/9+XdK6m8KjLIB0VlapOv/9AAL7z72P2ICLJNaoU1bBHQmCM1p85DqYBSoTA5lWjos8cWif5uNX0JALt378YXX3yBP//8E6tXr0bdunXxww8/IDo6Gt26dbN1H8nJbLFM2RVZmiN3MCMHtyUOj5/JyreqZg6RvQX7eUKpVFo8ja4AoHThEXL9uoH2sCUtC9/uvVg5vUNCEeecwlJEBPm6xBdaWwQ0xliSK2fJZ48tVp868+kse/Xlzz//jISEBPj5+eHo0aMoLq540uXl5XH0rIqyxTJle7JkRaiUrTzK1QL2nr+F//x+Bv/5/Sz2XriFcrUga7n0FYnTnETO8v6TsRjWpaHFjxcA3C1x3aLhYkDW4z77VWlfm3LVqtFwV/lCK2W7uAALiqtamitnyWePLVafxjVy8Yr+2t5//30sXrwYw4YNw4oVKzS3d+3aFe+//75NO0euwdpCrPZkarTLkhpjYp7ClDUnMfnnE8i9908S/YLtFxDg7SF5dwsAaBhqPn+CyJleWZlik5GkYD8vm0z7BPp62mW/1pNXpW8XKEdogLfVi3Wc9YVWn5T6jv96MBrzt16QdVxLc+Us/ewxWmQ3yEfS6GVHJ3yWiWQHZWfPnkX37t0r3a5SqZCbm2uLPpGLsUUhVlsrKVPj32tO4qejf1W6LzOvCGN+PIpgfy+dVThisKby8zabI2dserJQxohARJAPno+Lwtd7MjiFSQ6nAOCpVKDUzMixrab2RnSNxry/twLTf48QIG0aMSLIB7ve6oUjl25XVMYP8MHrq4/jer51hVCBioU1If5eklMPpOoUFYKkU5Yt2HHmF1pjHo6JwGvx92HJ3gydIDtC68vu98mXTK5wVPl54vMh7XGroPKOBnJY89ljaPVpWbkaz39rvp7qoYs56Oqk/S9lT19GRETgwoXKUfKePXvQqFEjm3SKXI/UQqyOKC47a1Mamr/7m8GATJv+m4YYrP0vOcPmfTJk6mMx8PZU4t3EFg45H5E2AUCpWkCIvxf8vWS/1csSGuCFcb2aGH2PmBDfVFLwN7hTA3h7KjWFUbs2DcP0J6wvhCrq37auDY6iy9fLsr0SnfWF1pSk1Ex0+3gb5m45pwnIgv28MCG+KfZM6oU+sZHwUCo0G6sb8/GAVujaJMxgcVu5rCkCrl9k94CEYt4AbLpVlVyyR8pGjRqF8ePH49tvv4VCocC1a9eQnJyMN954A++++649+kguwlwhVkcUl521KQ1f7LIuqHJUGYrzNwoAACEBPg45H5Ehth4ZMuSpNnXhoVQYfY9Yf/yapONEhQVUuk1q3Sop4mMi0DE6FJPXnLRZLas6wX5mp9hU/l7w9fTQ2fPS1uUvrGWsdE/evVLM23IezSL+qcfWJzYSi4e2w/T1p3SmAyOCfDD9CcPbQ1nDVkXAy9TSZjqktrMH2UHZ5MmToVar0bt3b9y9exfdu3eHj48P3njjDbzyyiv26CO5EGP1bRxRXLakTI2vdjtmlMsW5m45j6bhgShVV79SAVS91An2w7qUq5oPS+33iKTUTMzccErScYzlVul/KGtv+B0W4IOJq1Jw/Y7pXCFxax8PpUKzX2zyn7cAVLyn5d0txcyN8gO/rk3C0KqeyuQU20f9W9plZxFbsaQWpb12SzHGFkXAC4qkBVtS29mD7KBMoVDg7bffxptvvokLFy6goKAAMTExqFGjhj36R27A3AsaAN5em4pezWvD29PyaZQfkitv/eHqxi47iv7tpO+LR+RulApg5sbTmt+1R8eNfVnTJyW3ytSH8own78eYH4+aPIf2NKGHUoGuTcPQtalu3lBCrG7gN+p/h3C3xPiXqhB/L3RpVDE9ZzCxXG80zBXKXhhiaS1Ke+yWYk9qiQmUUtvZg+xPyBdffBF37tyBt7c3YmJi0KlTJ9SoUQOFhYV48cUX7dFHcnFSNufOLixBl1lbNBvaylWuFiTnA7gSAcDPR6VN3RC5I2N7JG46kWn0y5o2W+RWidNphjaiDvGv2DRcyki9dg7SnaJSkwEZAMzq31Jn5GjPpF5YPqoL5g9qg+WjumjysFxdVa1Fqe+mmdFUue3sQXZQ9v333+Pevcq1l+7du4f//e9/NumUIbNmzULHjh0RGBiI8PBw9OvXD2fPntVpU1RUhLFjx6JmzZqoUaMGBgwYgOvXdfOHLl++jMTERPj7+yM8PBxvvvkmysp0l1/v2LED7dq1g4+PD5o0aYLvvvuuUn8WLlyIqKgo+Pr6onPnzjh4UHdFh5S+VBVSX6g5haWaGmByiMmnf6RVzetH5I6MxU9iEPbuulRJU4GhAd42SW/oExuJI+88jKUjO2PcQ40x7qEmWPqvzjj8zsOyjy2O/psS4u+Fh2MidG7TTyx3xPSkLRZXuXotSlupFSgtv1dqO3uQHJTl5+cjLy8PgiDgzp07yM/P1/zcvn0bmzZtQnh4uN06unPnTowdOxb79+/H5s2bUVpaikceeQSFhYWaNhMmTMCvv/6K1atXY+fOnbh27Rr69++vub+8vByJiYkoKSnBvn378P333+O7777D1KlTNW0yMjKQmJiIhx56CCkpKXjttdfwr3/9C7///rumzcqVKzFx4kRMmzYNR48eRevWrZGQkIAbN25I7ktVIveFOuPXNMlvHMaKvBKRc4x7qAneTWxhMpVAQMXouBTvJLaw6V6PXZuG4Y2E5ngjoRm6NgmzKDCSMvp/+24pDjp59F78wjr4q/0YvyIFg7/aj24fb5P9xVdK0dhIFyvdYQmpzwVn5vopBEHa5KlSqTRZNFOhUGDGjBl4++23bdY5U27evInw8HDs3LkT3bt3R15eHmrVqoVly5bh6aefBgCcOXMGLVq0QHJyMrp06YLffvsNjz32GK5du4batWsDABYvXoxJkybh5s2b8Pb2xqRJk7Bx40akpqZqzjVo0CDk5uYiKSkJANC5c2d07NgRCxYsAACo1WrUr18fr7zyCiZPniypL+bk5+dDpVIhLy8PQUFBNr12tlauFtDt422ytrVYPqqL2VwE8bgMyIicT8z72jOpFzacuIbxK1Jsclwp7wWOti7lqqS/b/6gNniyje3LbEhhLF9P/JSWO/ooHg8wvFjBFqOZzvbz4St4/acTZtt98nQrDOhQ3+LzWPP5LXmkbPv27di6dSsEQcBPP/2Ebdu2aX727NmDy5cvOywgAyq2dQKA0NCKyP3IkSMoLS1FfHy8pk3z5s3RoEEDJCcnAwCSk5PRsmVLTUAGAAkJCcjPz8epU6c0bbSPIbYRj1FSUoIjR47otFEqlYiPj9e0kdIXfcXFxTqjj/n59qk+bQ/a21pIJWXKU8q3VSJyHDHvS+roeGiAl8n6YkoFcNvKavj2IPXvu3ir0HwjO5CyuErOjARgXT0wd1EnRNoOK1Lb2YPk1Zc9evQAUDG916BBA1lbzdiaWq3Ga6+9hq5duyI2NhYAkJWVBW9vbwQHB+u0rV27NrKysjRttAMy8X7xPlNt8vPzce/ePdy+fRvl5eUG25w5c0ZyX/TNmjULM2bMkHgFXI/4gv732lRJW45IedNz96RSoqpCqQAWDG6r+WCWuv3Nu4ktMHbZMaPHVQsVK5QXKS3/0C9XCzYvyyB1U+u5evW7HMXS1ZLmOLrMhaOJ/66mrp2zp2llJ/pv27YNP/30U6XbV69eje+//94mnTJn7NixSE1N1dl7091NmTIFeXl5mp8rV644u0uylKsFqPy88fajzRHoazzWl5Ob4O5JpUSuqH/bSAT7VV6laIpa0C2CbGrTZ+3VlH1b1cHC59oaXRQg0h7VkZO4bqucKn1SR//F+l322LnEFHuulpSzWMERO7jYkvjvaip3ztk7LMiuUzZr1ix88cUXlW4PDw/H6NGjMXz4cJt0zJhx48Zhw4YN2LVrF+rVq6e5PSIiAiUlJcjNzdUZobp+/ToiIiI0bfRXSYorIrXb6K+SvH79OoKCguDn5wcPDw94eHgYbKN9DHN90efj4wMfH/es/G6okr8hcpe+d4oOlbyBLBGZ5+/tgTnPtAUAzWjI+esFWLDd/AbT2tXoARObPuvV5goJ8DG7KEAc1cm7VyJ5VxBjOVXidmqfP9cWfVtZXiOwT2wkXou/D3P/3s/TXN8dmRfnCqslHbGDiz0Ye966St9lj5RdvnwZ0dHRlW5v2LAhLl++bJNOGSIIAsaNG4e1a9di27ZtlfrQvn17eHl5YevWrZrbzp49i8uXLyMuLg4AEBcXh5MnT+qskty8eTOCgoIQExOjaaN9DLGNeAxvb2+0b99ep41arcbWrVs1baT0paqQszpSbm6Ch1KBwZ0aWNtFIvrbp8+2hodSoTMaInXj5ZkbTlUagZJSm0vqaM2WtCyD7yVi3TPtc5vKqRKNW34Mm05YN2IWFSYtt8jRqRbOXi1p7H3f0L+VK3LlmnKyR8rCw8Nx4sQJREVF6dx+/Phx1Kxpv28KY8eOxbJly7Bu3ToEBgZqcrNUKhX8/PygUqkwcuRITJw4EaGhoQgKCsIrr7yCuLg4zWrHRx55BDExMXj++ecxe/ZsZGVl4Z133sHYsWM1o1RjxozBggUL8NZbb+HFF1/Etm3bsGrVKmzcuFHTl4kTJ2L48OHo0KEDOnXqhHnz5qGwsBAjRozQ9MlcX6oCKW+MoQFeePex+xERZFlugqG98IhIntqB3pjxZKzBDx2p+VNinUH9L1bmqrpLHa1Zm3JV8jY/UhYBqQXg/5YdxWIr8tVcYUTKEHEaztTWTvaahrNkSyZX5Kq7EcgOygYPHoxXX30VgYGB6N69O4CKGmLjx4/HoEGDbN5B0aJFiwAAPXv21Ll9yZIleOGFFwAAc+fOhVKpxIABA1BcXIyEhAR8/vnnmrYeHh7YsGEDXn75ZcTFxSEgIADDhw/He++9p2kTHR2NjRs3YsKECZg/fz7q1auHr7/+GgkJCZo2AwcOxM2bNzF16lRkZWWhTZs2SEpK0kn+N9eXqkDKG2NOYSkignx1nvxyEnOZV0YknwIVRVnfSWyBCJWfydeYqQ94Q+R+4EpZFBAa4G2yrpn+NGFWXuUC5rbqr9y+m9seyl6kTh/bmr0WGVAFyXXKRCUlJXj++eexevVqeHpWxHRqtRrDhg3D4sWL4e3tbZeOVjfuUKfMklo+cvMQSsrUaP7ub2635yWRs1hTp0rq6ul3E1vgha7RkgMdczWwRnSNwrd7L5o9jvhe8s3uP3X22zTHmlpo1tbvssfqUEceX5871HBzNms+v2WPlHl7e2PlypWYOXMmjh8/Dj8/P7Rs2RINGzaUeyhyc3KH9o0l5op5CIbe3I5cus2AjMgIhQJQ+Xoi994/W8VZOlLSJzYS90rKMWHVcbNtZ248ja/3ZEg+j7lRHZWft6SgTHwvCa0hb1GUNTlf1oxIOSIZ3tHTcK46pVtVyA7KRPfddx/uu+8+W/aF3IycoX2peQi9mtfGkUu3Nd/65ExTEFU3ggAsHNIeSoXCJiMlESo/yW1NfZkyRKyBtf/PbCSnZwMQENcoDF3+DijkTBNGBMn7wLc2QLCkfpclX0LdgStP6VYFkoKyiRMnYubMmQgICMDEiRNNtv30009t0jFyfWIuypi/h/a16SebJqdnS8pDaDfzDxQUl2tuN1XzjKq3Gj4eOs+V6upWQbHNpomkJv0DliV1b07L0hk5WrA9XTNyJCdxXUoRUJGtViHKGZGqKsnwhjhzkUF1IKkkxrFjx1BaWqr5f2M/KSkp9uwruahg/8qFKFX+XjrfBKVOH+h/yN4pKjPSkqozf2/3DMjEpPYgG37ZsOU0kamisIZoJ3WbY66MAgDJ2/x4KBV4orX5USZnFQOVkwzvjqrDlkzOIumdYfv27Qb/n6o3Y8PzAJB3t1Tnd+YXkC3dLXHPgAwAPnyqYms4QyPMco9nj2ki8QN3+vpTkgs3m/vSJXXkaM+kXpKmCcvVAtYfN10LS39rKEeyZ8V9V1HVt2RyFs4NkUWk1CjTHp6XMy1CVBXpJ4YvHtoOk9ecRK7eFxgpHDNNJP24Z7PuIDk92+iHstwyCuamCaXWKdPeGsqRqksyvKvW+nJnkoKy/v37Sz7gmjVrLO4MuQ+5b7JyayERVSWGSkiIIw37LtzCy0uPyJqOVfl5YUTXKDwcY3jbNmuYGgE35vMd6fh8R7rRlYVSR4SkLuxx9ZEoJsO7PkeXEpFKUk6ZSqXS/AQFBWHr1q04fPiw5v4jR45g69atUKlUduso2YatNpC15E3RWB4Ckat7qo3leygCwKWcuziYkVPp9eahVODB+2ph9oBWZo+hAODvVfGWnXuvFHO3nLfJ5tvaSsrU+PfaVIu/NGUa2WZH6ojQzI2nJf09rj4SJXXDdlcIAlyVPTc7t9dG9rYgu3jspEmTkJOTg8WLF8PDwwMAUF5ejv/7v/9DUFAQ5syZY5eOVjf2KB5ry5o5yenZGPzVfrPtDBVtFL+h7L1wEwu2p8s6L5EzLP1XZ7yx+rjV0+/GXm9SX0/6LC0Ua0hF8diTyCmUP52qL1Lliz2TemmCjnK1gG4fbzN7/eQUZDV1PHEkSrsPzuCum3Y7W1JqZqWcxoggH0x/4n6bPM8NjQTb8rVkzee37KCsVq1a2LNnD5o1a6Zz+9mzZ/HAAw8gOztbVgfIMFsHZbZ+ItriTVHqGzWRM4kBxua/N8wGLJ9+N/Z6k1ol3dgxrQ1ALJmyNEf/C5nUc0j9e6yttO8orjpN5qqSUjNNLoJZbMW/q/iZYyz1xlbBvDWf35KmL7WVlZXhzJkzlW4/c+YM1Gq13MORA5hb+QRUJOXLGR7WHp43RADwROtInSe2oeHoQR0bMCAjlyZOM9li+t3Y682aaTZryytIWbRjCf0UB/H6hQZULqGjTerf4y5lGcRk+Cfb1NXk15Jh5WoBk9ecNNlm8pqTFk9lukOpEtmrL0eMGIGRI0ciPT0dnTp1AgAcOHAAH330EUaMGGHzDpL17LWBbJ/YSIzuHo0vdmUYvP/LXRlo2yAEfWIjDQ7jKxUwuoWSj6cSxWUM8sm5gv08dZLp9csAnL9egAXbL8g6pqHXmy1WJ1ua1C5lJaMlDAWafWIjca9UjQkrU8w+Xsrf0yc2Ej3uC8eHm9JwMfsuomr64999Y+Dn7WFJl8nJ9qdnm12NnHu3FPvTs9G1aZjs47v6AhHAgqDsP//5DyIiIvDJJ58gM7MiKS4yMhJvvvkmXn/9dZt3kKxnryeilFpBM35Ng1oNjF1WedrCWED2Wu+m+L+HmqDjB1uQd8/6/BZyTwpUFCH29fRAVr5z3iRz75VV+rKiXQYgOT1bdlAm0n692WJ1sqWjbfb4ADJVRV/qFklS/p5Zm9Lw1e4MzXvJ7vPA0gOXMerBaEzpa3wkn1xT8p+3JLezJChz9QUigAVBmVKpxFtvvYW33noL+fn5AGCzRHSyD3s9EaWOwL2zTt5qrs+2nUejmv5oU1+FneekvUipahEneD7q31JnZCoswAeHLmbju32XkOuggH1LWpbREWRrRrj0X2/GNr6WIjTAy+LyClJf9zUDvPF0+7r4cleG2WR9UysLbVUuYtamNIOj9GoBmtsZmLkbqVO7lk0Bu0OpEtk5ZUBFXtmWLVuwfPlyKBQVF+fatWsoKCiwaefINsQnorGnsQKW7Q8n9Rt2TmGJrOOqBeDVVccZkFUjIXpbdWnnBWnn5NwpLsX8rRccFpABwNqUq0ZzWORuSyS2M/Z66xMbiT2TemH5qC6YP6gN+sbWlnTMp9rUtThXydz7A1AR9CVP6Y0pfWOwaGg7RBrJq4uUkM9li3IRJWVqfLXbcNqE6KvdGShhCoRbkZo+Y2nBWncoVSJ7pOzSpUvo06cPLl++jOLiYjz88MMIDAzExx9/jOLiYixevNge/SQr2GsDWXevRk2uw9tDgQnxTREVFlBphZq4ei0rvwgzN5xy+MKQnMJSk/mWcka4pLzexCC0IgH/lKQ+xltRRFbK+8OHT7WEh1KB5PRsFJep8Z+nWwMK4MadYuQUFCM0wBsRKj/JKwuNXTP9XQ+M+SH5otH0B5FaqGg38sFGZvtDrqFLo5oI9vcymVcW4u+FLo0s30XA2ueevckOysaPH48OHTrg+PHjqFnznwvz1FNPYdSoUTbtHNmOPZ6IUoaCQwK8bFL3iKq2G3dKMG/LeSwa2q5SGQVLpvNszdyosKF9AG8XFmPmxtMWv94OZuRIeu3UDPC2errF3PsDgEqlBMR6W5YGPdbsnXgp566kc0htR67BQ6nAR/1bmiyJMat/S6tHslx5307ZQdnu3buxb98+eHt769weFRWFq1ev2qxjZHu2fiJK+Yb9/pOxlT6YiPSJm1JPX38Kgb5euFVQjIu3CjF3y3lndw2AtFFhQ/sAJsRGWvx6k5oe8GSbOjb5MDH2/iDWZ9P/4pX1d/V+a8pPWLp3YsNQf5u2I9fRJzYSi4e2w/T1aToLfGxddNdV9+2UHZSp1WqUl1feo+2vv/5CYGCgTTpF9mPrJ6KUEbjjf+UaLZtBJBIAZOUXY8jXB5zdFQ1rE3+teb1JTQ+w5f6X+v01V+NQgYoV1g/HRJgMDG1dQPX5uCh8sOm0ySlMpaKiHbkfVx7JsjfZQdkjjzyCefPm4csvvwQAKBQKFBQUYNq0aejbt6/NO0iuz9QLSErZDCJbC/L1xIB2dbHu+DWLp8+dnfgrZWWnJQt0jDEUONmixqE9thry9lRi1IPGayQCwKgHo+HtadFaNnIBrjqSZW8W1Snr06cPYmJiUFRUhOeeew7nz59HWFgYli9fbo8+khsw9gLan57NqUtyuPyiMjxyfyTeeex+fPLHWXy+w/weqypfT+QVlWl+d3bir3Z6gCHmSk/IYSxwejRW2iicsalWY1sr2WLqUyx3oV+iQwFgdHfWKXMEbiFle7KDsvr16+P48eNYuXIljh8/joKCAowcORJDhgyBn5+fPfpIbiopNROTfja9ZQaRvVzLvQcPpQI1A7zNNwYwrldTxNZVudQHjLhrhnaBVKBiam7Ug9E2CRhNBU7f7r0o6RiGplptNfVpStsGIagdlKmTe1Q7yBdtG4RYdDySjput24esoKy0tBTNmzfHhg0bMGTIEAwZMsRe/SI3Z25TWSJ7e+On4zh3PR/NI1WS2ocF+rjcdElSaqbBYq2CoLuNmaWkBE4KE9uhmcq5s9f2biJjweT1fOtH4cg0e46AVneyJty9vLxQVMSpKDKuXC1g7/lbmLjquLO7QtWc8Hdl922nsyS1l7r9j6OYC5iAyhubyyUlcBIPL7fYpj33GXTEtSHDeO3tS3YW5NixY/Hxxx+jrKzMfGOqVpJSM9Ht420Y8s0B3C2pvEKXSNu4h5rgxa5Rdj/PhhNZZgMuWybM24qckSZLSQ2IRnaNQoReFf8IM9X77bnPoCOuDRnGa29fsnPKDh06hK1bt+KPP/5Ay5YtERAQoHP/mjVrbNY5ch/GhrOJjIlrVBNv/CRtRHVC/H2ICvPH+et3sGC7+aR9bQKA6JoBRjc1t2XCvC3Zc6RJJDUgio+JwL8TY2Qlddtzn0FHXBsyjNfevmQHZcHBwRgwYIA9+kJuwNBqGwBGh7OJDFEqALUgSFqZOyG+KcbHNwUAJKdnyw7KACA5I9vofaO72yZh3tbsOdIkkhM4yS1RYK/t3QDHXBsyjNfevmQHZUuWLLFHP8gNGFttM6hjA5a9IFnUAnBA4vRGVNg/o/FSanfJterwX3irTwuXGymz50iTyJ6BE2C/fQYdcW3IMPHam3rPd8V0AHchOadMrVbj448/RteuXdGxY0dMnjwZ9+7ds2ffyIWI05P6L8SsvCLM3XLOSb0iZ1KgYnPgiCAfnduD/bwkHkFaWKX9jVsMImzp9t1S7P/T+Eias2j/rXKT7OUQAye5OWNyjr9nUi8sH9UF8we1wfJRXbBnUi+rjuuoa0OVeSgVeKK16X+7J1pH8tpbSCEIgqR3xpkzZ2L69OmIj4+Hn58ffv/9dwwePBjffvutvftYLeXn50OlUiEvLw9BQUFO7Uu5Wqi0GTGRaOm/OkOpUGimtNWCIGmrpKUjO+ONn46bHe3YM6lXpTf4pNRM/HvtSZttdj/uoSZ4I6GZTY5la46qB+WOhUBZK8vxpHweRBp53VYX1nx+S56+/N///ofPP/8cL730EgBgy5YtSExMxNdffw2lkltZVGXmVttQ9XaroBhPtqmr+b1cLUiaWurSuKbFU2d9YiPRq3ltdJm1FTmFJTb4K1w3I9JR+wC647Y21XmPRGeR8nlgTf256k5yNHX58mWdvS3j4+OhUChw7do1u3SMXAdX0ZAp+gm9cqaWrJk68/ZU4sOnYisKnFr5N8Q1CrPyCPYlBkxPtqmLuMY1GXRo4bVxLK6+tC/JI2VlZWXw9dV94/Ty8kJpqW2mD8h1cRUNGWMsoVdOgrc1ox3GziNHsL8XuvAbPZEkXH1pX5KDMkEQ8MILL8DH55+k3qKiIowZM0anVhnrlFU9UlbbUPUkjngZykeSE2xZM3Wmfx65tcw+6t+SoytuzB1z4dwZV77al+SgbPjw4ZVuGzp0qE07Q65JXG3zxa4MZ3eFXMjIrlHoExtpNtnaEXkl2kGd1FpmIf5emNW/JRPC3RgT/R3P3mVUqjvJqy/Jsbj6svrw91Tgbpn7vQyXj+qCvHslBndyEN+OnbExsfh8NVXLrGaAN5Kn9Ia3JxcpuStju4g487lXnTAgNs4hqy+p+uLqS/tyt4BMnJ5o3zAEPeZsN7oxsQIVOz08HBNh8bdmS6ampHyT/+CpWAZkbszcpti2eO6RaVz5ah8MysgsrqJxDE8lUKZ2di9M056eOHLptuSNiS2ZwpTyTdxY0GavSvLkGuRsis2yDPbjjmVUXB2DMjKLq2gcw9UDMgBQ+Xvho7/zsNalXJX0GEuCemNTU1l5RXj5x6NYNLQdAJgM2vhNvupiWQaqqhiUkVn22G+QbKuGjycKisvsfh4/Lw88HBMBwH5L46VMTU1ZcxK371Yux6MdtPWJjeQ3+SqKZRmoqmJSBZkl5ugwIHNNCgD/eaYVFg9th0iVfT+ExCkh4J9g3di4kwKWbUwsZWrKUEAm3gdUjKCVq/mMrars9dwjcjYGZSRJn9hITIhv6uxukJ4gX0+M6BoFlZ83Ho6J0Gz8PLJrFEIDpG4MLk9WfkXAZK9Noa2dctLOJ6KqiRuSU1XFoIwkiwoLMN+IHEYBIL+oDN/uvYjBX+1Hxw+24PfULMQ1rol3H78fh95+GMtHdUGv5rVset6ZG04hKTUTAKzaJskYW005MZ+oarPHc4/I2VinzEW5Up0yUXJ6NgZ/td/Z3SAzXuoejSl9Y3Ru23TiGt5Zl4qcQttsi6aAbh0oW1ZVl1JnTIrlo7own6waYEV/cjXWfH4zKHNRrhiUlasFtJ+5Gbn33Hu/U39vD9wtKXd2NyRTKgDt9Cj93w35/Lm26Nuqjs5t2h9eF28VYu6W85XqeEkl1irbM6mXXT4AxdWXgOE6Yyp/L+TdLTW5zYu9+kZEZIo1n9+cviTJPJQKjOga7exuWCU0wBv/ebo1JsTf5+yuSBbkq5sbJiV//Z11qZUS3cWViE+2qYvx8fdhsYGpn0iVL17qHo3QAG+Tx7d33pa5qamP+rcEwHwiIqpaWBKDJBFHWRrU9EeAjwcKi91npElbTmEJxi47ioXPtUWIv5fRVXyuIMDbA4Ul5RaNTOYUlpotnGmqjlfziCBMWHXc7Hnsmbdlrs4Yi8MSUVXDoIzMMlRZ3Z0JAGZuPI33nozFK8uPObs7Rnl7KlFoxTSrlIDJWB2vCJWfpHPYuw6UqTpjLA5LRFUNgzIyyVhldXeXmVeEsBo+eKl7NL7YleHs7uhQoGKaNbuwxKrjWBMwmSsYLOZtObsOFIvDElFVwpwyMspUZXXgn+AhyNc9Y/usvHt4q08LBHh72OyYSkXFQgJLiWM8T7apY7KdOdYWzjRVBwqoGG0c1LG+xccnIqLKGJSRUVIqq+cUlqDcTRfw5hSWYP+f2VZNEYr8vTzwdt8W+H5EJ6tWdoqJ7OJWRpZQwDaJ7saS7UVzt5xHt4+3aWqWERGRdRiUkVFSk7jdNek/tIYPktOzbXKsu6XliK2rQs5dy6cc301sgT2TeqFPbKTZbWSMCfb3smnhzD6xkdgzqZfR1ariXpMMzIiIrMegjIyq6pv5Xs4uhGVVugwTk80tFRbooxndMjd9aIz2huG2tOLQZYO3c69JIiLbYVBGRlk6WuMulh+8jM5RtksSF1f/RQT5WPx4beamDw2xR+0wKdPY3GuSiMh6DMrIKEtHa9xFVn4xlB4KqxLzgYprIybWeygVGNypgcWP1ydOHy4f1QXD4hpKOp6ta4dJPR73miQisg6DMjLJ2GhNTTMV393FrYJivNS9seT2UirIy9m4XUoFerHsw6MS88TkTKGWqwUkp2djXcpVJKdnG5yClHq8qj7dTURkb+5Zy4AcylCRzvYNQ9BjznarN412trAAHzwaG4kvd6ebXLAQ7O+FD/vFYubG02YryMsJTuRUoLd17TBDRYEjDfTHXWqWERG5O25I7qJccUNyfcY2jXYnr/ZqgtVH/jK7W8Hiv1c0am/qbayCfLlaQLePt5kMWIP9vbBwcDt0aVxTVukKcxt1S115aawosLHj2Oq8RERVnTWf3wzKXJQ7BGVA1duCSZ+hkSMp7BnESB3hMkYMGo39m4kjX3sm9dIJGK09LxFRdcCgrApyl6AMgM7oUViAD15ffRzX8917WhOoyJtLntIb3p6WpV4aC2LeTWyBkAAfq/ZrlDJiZ0xyejYGf7XfbLvlo7pU2sLImvMSEVUH1nx+M6eMrKa//+D0J2Lw8o9HoYD7TmsCQHZhCY5cuo1O0aGVAhEAZoMTQ7l4twtLMHOj9aNN1uz5aM1qSu41SURkPwzKyObEFZv6o0Qh/l4oLC5DSbn7hGqb07IwcVWKzt8R7O8FAMi9W6q5zVhgpR3EJKVmYuyyynlcYlV8R+VlcTUlEVV3rjrqz+lLF+VO05fGGHrSb07Lwpi/c62qEnO5YpbmcdmDuYUIjuyLI7nqmzAROZa982Ot+fxmnTI7W7hwIaKiouDr64vOnTvj4MGDzu6Sw4ijRE+2qYu4v1cZ9omNxOKh7SpVvY8I8rG6iKutyfm8NrfdkCtVxTdVFFhK3TR3lJSaiW4fb8Pgr/Zj/IoUDP5qPzdTJ6qGxEVY+u/HrrKPL4MyO1q5ciUmTpyIadOm4ejRo2jdujUSEhJw48YNZ3fNqfrERmLv5N5YPqoL5g9qg+WjumDXW70sTqi3F7lbOZoKrFytKr6xosARKt8qV97C1d+EicgxytUCZvyaZnCGwFX28WVOmR19+umnGDVqFEaMGAEAWLx4MTZu3Ihvv/0WkydPdnLvnEs/YTw5PVsnR8sa/l5K3C1V2+RYljAUWLliHpehhQhVbUrP3JuwAhVvwg/HRFSpv5uIKpMzY+GsBU2uNTRRhZSUlODIkSOIj4/X3KZUKhEfH4/k5GQn9sw12XKE6KUejZ26V6ehwMrc5u6m9r+0J0NTzFWJK00bE5FzudqMhSEMyuzk1q1bKC8vR+3atXVur127NrKysiq1Ly4uRn5+vs5PdWKrEaJIlS/G9WqKhc+1lZUTZgvmAqtBHRsYTawHql4elytwhzdhInIMV5yx0MegzEXMmjULKpVK81O/fn1nd8mhzI0kSaHAP4FNSICP7JwwU8eV2sZQYCUmmc/dcs7gY6tiHpercIc3YSJyDFedsdDGoMxOwsLC4OHhgevXr+vcfv36dURERFRqP2XKFOTl5Wl+rly54qiuugRTKwKlCPbz0glsbDXyMSH+vkrJ8CH+XppaZSJjgZWxJPN/jt8Ueyb1YkBmJ+7wJkxEjuEOK8+Z6G8n3t7eaN++PbZu3Yp+/foBANRqNbZu3Ypx48ZVau/j4wMfH59Kt7syW9d9MlZ0VoqFQ9qha5Mwze/WjnyItbrG9WqCcb2aWFTR31SSuXiOFYeuYFyvplb1lYwT34QN7TDhKm/CROQ4xj5nIlxkH18GZXY0ceJEDB8+HB06dECnTp0wb948FBYWalZjujN7Fd/TXxH4581CzN963uRjIlW+6NJId6WMOEJiqkBqsL8Xbt8tlfRhbWgljrnVOe6w0qc6cPU3YSJyLFdeec6gzI4GDhyImzdvYurUqcjKykKbNm2QlJRUKfnf3YhTcrbaLsjQiJt2kFJUWoYvdmUYfKx2Hpk2KSMks/q3BAC7fVgzydx1uPKbMBE5nqvu48ttllyUq26zZOvtgqSOuG06kYl31qUip7DEZDtLjm+v7XeS07Mx+Kv9ZtstH9XF5JsDtwciInIf1nx+MyhzUa4alNkq0ACMj7gZ20fS0uDEWUGNLfaYtPcebUREZFvc+5IcxlZTcpZsd2FpoVNnFUi1dqUPtwciIqpeGJSRLLaq+1RdKq1busekO+zRRkREtsVEf5JFyqrGCAl1n6pTErwlSeZcuUlEVP0wKCNZbFX3qbpVWpe70qc6Ba1ERFSB05ckm6VTctpYad206ha0EhERR8rIQtbWfWKlddNsNU1MRETugyNlZDFrVzXaYsStqnKHPdqIiMi2WKfMRblqnTJ7YHFU41injIjIvbB4bBVUnYIyMo1BKxGR+7Dm85s5ZUQuzlX3aKPqhV8OiOyPQRkREZnEaXQix2CiPxERGcXtvogch0EZEREZxO2+iByLQRkRkYsrVwtITs/GupSrSE7PdlgQVF32qCVyFcwpI5fHBGOqzpyZz8Xtvogci0EZuTQmGFN1JuZz6Y+Liflc9i6yzO2+iByL05fksphgTNWZK+RzcY9aIsdiUEYuyRU+kIicyRXyubjdF5FjMSgjl+QKH0hEzuQq+Vzco5bIcZhTRi7JVT6QiJzFlfK5+sRG4uGYCC64IbIzBmXkklzpA4nIGcR8rqy8IoPT+ApUjFY5Kp+L230R2R+nL8klMcGYqjvmcxFVPwzKyCXxA4mI+VxE1Y1CEAQuX3NB+fn5UKlUyMvLQ1BQkLO74zSsU0bEAspE7sSaz28GZS6KQdk/+IFERETuwprPbyb6k8tjgjEREVUHzCkjIiIicgEMyoiIiIhcAIMyIiIiIhfAoIyIiIjIBTAoIyIiInIBDMqIiIiIXACDMiIiIiIXwKCMiIiIyAUwKCMiIiJyAQzKiIiIiFwAgzIiIiIiF8CgjIiIiMgFMCgjIiIicgEMyoiIiIhcAIMyIiIiIhfAoIyIiIjIBTAoIyIiInIBDMqIiIiIXACDMiIiIiIXwKCMiIiIyAUwKCMiIiJyAQzKiIiIiFwAgzIiIiIiF8CgjIiIiMgFMCgjIiIicgEMyoiIiIhcAIMyIiIiIhfAoIyIiIjIBTAoIyIiInIBDMqIiIiIXACDMiIiIiIXwKCMiIiIyAUwKCMiIiJyAQzKiIiIiFwAgzIiIiIiF8CgjIiIiMgFMCgjIiIicgFuEZRdvHgRI0eORHR0NPz8/NC4cWNMmzYNJSUlOu1OnDiBBx98EL6+vqhfvz5mz55d6VirV69G8+bN4evri5YtW2LTpk069wuCgKlTpyIyMhJ+fn6Ij4/H+fPnddrk5ORgyJAhCAoKQnBwMEaOHImCggLZfSEiIiISuUVQdubMGajVanzxxRc4deoU5s6di8WLF+Pf//63pk1+fj4eeeQRNGzYEEeOHMGcOXMwffp0fPnll5o2+/btw+DBgzFy5EgcO3YM/fr1Q79+/ZCamqppM3v2bHz22WdYvHgxDhw4gICAACQkJKCoqEjTZsiQITh16hQ2b96MDRs2YNeuXRg9erSsvhARERHpENzU7NmzhejoaM3vn3/+uRASEiIUFxdrbps0aZLQrFkzze/PPvuskJiYqHOczp07Cy+99JIgCIKgVquFiIgIYc6cOZr7c3NzBR8fH2H58uWCIAhCWlqaAEA4dOiQps1vv/0mKBQK4erVq5L7Yk5eXp4AQMjLy5P8GKqaysrVwr4Lt4Rfjv0l7LtwSygrVzu7S0REZIQ1n99uMVJmSF5eHkJDQzW/Jycno3v37vD29tbclpCQgLNnz+L27duaNvHx8TrHSUhIQHJyMgAgIyMDWVlZOm1UKhU6d+6saZOcnIzg4GB06NBB0yY+Ph5KpRIHDhyQ3Bd9xcXFyM/P1/khSkrNRLePt2HwV/sxfkUKBn+1H90+3oak1Exnd42IiGzMLYOyCxcu4L///S9eeuklzW1ZWVmoXbu2Tjvx96ysLJNttO/XfpyxNuHh4Tr3e3p6IjQ01Ox5tM+hb9asWVCpVJqf+vXrm7oEVA0kpWbi5R+PIjOvSOf2rLwivPzjUQZmRERVjFODssmTJ0OhUJj8OXPmjM5jrl69ij59+uCZZ57BqFGjnNRz25syZQry8vI0P1euXHF2l8iJytUCZvyaBsHAfeJtM35NQ7naUAsiInJHns48+euvv44XXnjBZJtGjRpp/v/atWt46KGH8MADD1RKmo+IiMD169d1bhN/j4iIMNlG+37xtsjISJ02bdq00bS5ceOGzjHKysqQk5Nj9jza59Dn4+MDHx8fg/dR9XMwI6fSCJk2AUBmXhEOZuQgrnFNx3WMiIjsxqkjZbVq1ULz5s1N/oh5WVevXkXPnj3Rvn17LFmyBEqlbtfj4uKwa9culJaWam7bvHkzmjVrhpCQEE2brVu36jxu8+bNiIuLAwBER0cjIiJCp01+fj4OHDigaRMXF4fc3FwcOXJE02bbtm1Qq9Xo3Lmz5L4QmXLjjvGAzJJ2RETk+twip0wMyBo0aID//Oc/uHnzJrKysnTys5577jl4e3tj5MiROHXqFFauXIn58+dj4sSJmjbjx49HUlISPvnkE5w5cwbTp0/H4cOHMW7cOACAQqHAa6+9hvfffx/r16/HyZMnMWzYMNSpUwf9+vUDALRo0QJ9+vTBqFGjcPDgQezduxfjxo3DoEGDUKdOHcl9ITIlPNDXpu2IiMj1OXX6UqrNmzfjwoULuHDhAurVq6dznyBU5NSoVCr88ccfGDt2LNq3b4+wsDBMnTpVp37YAw88gGXLluGdd97Bv//9bzRt2hS//PILYmNjNW3eeustFBYWYvTo0cjNzUW3bt2QlJQEX99/PvyWLl2KcePGoXfv3lAqlRgwYAA+++wzzf1S+kJkSqfoUESqfJGVV2Qwr0wBIELli07RoQbuJSIid6QQxKiGXEp+fj5UKhXy8vIQFBTk7O6QE4irLwHoBGaKv/+7aGg79ImNrPQ4IiJyHms+v91i+pKoOuoTG4lFQ9shQqU7RRmh8mVARkRUBbnF9CVRddUnNhIPx0TgYEYObtwpQnhgxZSlh1Jh/sFERORWGJQRuTgPpYJlL4iIqgFOXxIRERG5AAZlRERERC6AQRkRERGRC2BQRkREROQCGJQRERERuQAGZUREREQugEEZERERkQtgUEZERETkAhiUEREREbkAVvR3UeI+8fn5+U7uCREREUklfm6Ln+NyMChzUXfu3AEA1K9f38k9ISIiIrnu3LkDlUol6zEKwZJQjuxOrVbj2rVrEAQBDRo0wJUrVxAUFOTsblUb+fn5qF+/Pq+7E/DaOwevu3PwujuPva69IAi4c+cO6tSpA6VSXpYYR8pclFKpRL169TTDoEFBQXzBOgGvu/Pw2jsHr7tz8Lo7jz2uvdwRMhET/YmIiIhcAIMyIiIiIhfAoMzF+fj4YNq0afDx8XF2V6oVXnfn4bV3Dl535+B1dx5XvPZM9CciIiJyARwpIyIiInIBDMqIiIiIXACDMiIiIiIXwKCMiIiIyAUwKHNhCxcuRFRUFHx9fdG5c2ccPHjQ2V1yGbt27cLjjz+OOnXqQKFQ4JdfftG5XxAETJ06FZGRkfDz80N8fDzOnz+v0yYnJwdDhgxBUFAQgoODMXLkSBQUFOi0OXHiBB588EH4+vqifv36mD17dqW+rF69Gs2bN4evry9atmyJTZs2ye6Lu5g1axY6duyIwMBAhIeHo1+/fjh79qxOm6KiIowdOxY1a9ZEjRo1MGDAAFy/fl2nzeXLl5GYmAh/f3+Eh4fjzTffRFlZmU6bHTt2oF27dvDx8UGTJk3w3XffVeqPudeIlL64i0WLFqFVq1aaQpdxcXH47bffNPfzujvGRx99BIVCgddee01zG6+9fUyfPh0KhULnp3nz5pr7q+R1F8glrVixQvD29ha+/fZb4dSpU8KoUaOE4OBg4fr1687umkvYtGmT8Pbbbwtr1qwRAAhr167Vuf+jjz4SVCqV8MsvvwjHjx8XnnjiCSE6Olq4d++epk2fPn2E1q1bC/v37xd2794tNGnSRBg8eLDm/ry8PKF27drCkCFDhNTUVGH58uWCn5+f8MUXX2ja7N27V/Dw8BBmz54tpKWlCe+8847g5eUlnDx5UlZf3EVCQoKwZMkSITU1VUhJSRH69u0rNGjQQCgoKNC0GTNmjFC/fn1h69atwuHDh4UuXboIDzzwgOb+srIyITY2VoiPjxeOHTsmbNq0SQgLCxOmTJmiafPnn38K/v7+wsSJE4W0tDThv//9r+Dh4SEkJSVp2kh5jZjriztZv369sHHjRuHcuXPC2bNnhX//+9+Cl5eXkJqaKggCr7sjHDx4UIiKihJatWoljB8/XnM7r719TJs2Tbj//vuFzMxMzc/Nmzc191fF686gzEV16tRJGDt2rOb38vJyoU6dOsKsWbOc2CvXpB+UqdVqISIiQpgzZ47mttzcXMHHx0dYvny5IAiCkJaWJgAQDh06pGnz22+/CQqFQrh69aogCILw+eefCyEhIUJxcbGmzaRJk4RmzZppfn/22WeFxMREnf507txZeOmllyT3xZ3duHFDACDs3LlTEISKv83Ly0tYvXq1ps3p06cFAEJycrIgCBUBtVKpFLKysjRtFi1aJAQFBWmu9VtvvSXcf//9OucaOHCgkJCQoPnd3GtESl/cXUhIiPD111/zujvAnTt3hKZNmwqbN28WevTooQnKeO3tZ9q0aULr1q0N3ldVrzunL11QSUkJjhw5gvj4eM1tSqUS8fHxSE5OdmLP3ENGRgaysrJ0rp9KpULnzp011y85ORnBwcHo0KGDpk18fDyUSiUOHDigadO9e3d4e3tr2iQkJODs2bO4ffu2po32ecQ24nmk9MWd5eXlAQBCQ0MBAEeOHEFpaanO39u8eXM0aNBA59q3bNkStWvX1rRJSEhAfn4+Tp06pWlj6rpKeY1I6Yu7Ki8vx4oVK1BYWIi4uDhedwcYO3YsEhMTK10fXnv7On/+POrUqYNGjRphyJAhuHz5MoCqe90ZlLmgW7duoby8XOeJBAC1a9dGVlaWk3rlPsRrZOr6ZWVlITw8XOd+T09PhIaG6rQxdAztcxhro32/ub64K7Vajddeew1du3ZFbGwsgIq/19vbG8HBwTpt9a+Jpdc1Pz8f9+7dk/QakdIXd3Py5EnUqFEDPj4+GDNmDNauXYuYmBhedztbsWIFjh49ilmzZlW6j9fefjp37ozvvvsOSUlJWLRoETIyMvDggw/izp07Vfa6e8pqTUT0t7FjxyI1NRV79uxxdleqjWbNmiElJQV5eXn46aefMHz4cOzcudPZ3arSrly5gvHjx2Pz5s3w9fV1dneqlUcffVTz/61atULnzp3RsGFDrFq1Cn5+fk7smf1wpMwFhYWFwcPDo9LKjevXryMiIsJJvXIf4jUydf0iIiJw48YNnfvLysqQk5Oj08bQMbTPYayN9v3m+uKOxo0bhw0bNmD79u2oV6+e5vaIiAiUlJQgNzdXp73+NbH0ugYFBcHPz0/Sa0RKX9yNt7c3mjRpgvbt22PWrFlo3bo15s+fz+tuR0eOHMGNGzfQrl07eHp6wtPTEzt37sRnn30GT09P1K5dm9feQYKDg3HffffhwoULVfY5z6DMBXl7e6N9+/bYunWr5ja1Wo2tW7ciLi7OiT1zD9HR0YiIiNC5fvn5+Thw4IDm+sXFxSE3NxdHjhzRtNm2bRvUajU6d+6sabNr1y6UlpZq2mzevBnNmjVDSEiIpo32ecQ24nmk9MWdCIKAcePGYe3atdi2bRuio6N17m/fvj28vLx0/t6zZ8/i8uXLOtf+5MmTOkHx5s2bERQUhJiYGE0bU9dVymtESl/cnVqtRnFxMa+7HfXu3RsnT55ESkqK5qdDhw4YMmSI5v957R2joKAA6enpiIyMrLrPeVnLAshhVqxYIfj4+AjfffedkJaWJowePVoIDg7WWUVSnd25c0c4duyYcOzYMQGA8OmnnwrHjh0TLl26JAhCRRmK4OBgYd26dcKJEyeEJ5980mBJjLZt2woHDhwQ9uzZIzRt2lSnJEZubq5Qu3Zt4fnnnxdSU1OFFStWCP7+/pVKYnh6egr/+c9/hNOnTwvTpk0zWBLDXF/cxcsvvyyoVCphx44dOsvU7969q2kzZswYoUGDBsK2bduEw4cPC3FxcUJcXJzmfnGZ+iOPPCKkpKQISUlJQq1atQwuU3/zzTeF06dPCwsXLjS4TN3ca8RcX9zJ5MmThZ07dwoZGRnCiRMnhMmTJwsKhUL4448/BEHgdXck7dWXgsBrby+vv/66sGPHDiEjI0PYu3evEB8fL4SFhQk3btwQBKFqXncGZS7sv//9r9CgQQPB29tb6NSpk7B//35nd8llbN++XQBQ6Wf48OGCIFSUonj33XeF2rVrCz4+PkLv3r2Fs2fP6hwjOztbGDx4sFCjRg0hKChIGDFihHDnzh2dNsePHxe6desm+Pj4CHXr1hU++uijSn1ZtWqVcN999wne3t7C/fffL2zcuFHnfil9cReGrjkAYcmSJZo29+7dE/7v//5PCAkJEfz9/YWnnnpKyMzM1DnOxYsXhUcffVTw8/MTwsLChNdff10oLS3VabN9+3ahTZs2gre3t9CoUSOdc4jMvUak9MVdvPjii0LDhg0Fb29voVatWkLv3r01AZkg8Lo7kn5QxmtvHwMHDhQiIyMFb29voW7dusLAgQOFCxcuaO6vitddIQiCIG9sjYiIiIhsjTllRERERC6AQRkRERGRC2BQRkREROQCGJQRERERuQAGZUREREQugEEZERERkQtgUEZERETkAhiUERHZmUKhwC+//GLXc/Ts2ROvvfaaXc9BRPbFoIyIqozk5GR4eHggMTFR9mOjoqIwb94823fKjMcffxx9+vQxeN/u3buhUChw4sQJB/eKiJyBQRkRVRnffPMNXnnlFezatQvXrl1zdnckGTlyJDZv3oy//vqr0n1LlixBhw4d0KpVKyf0jIgcjUEZEVUJBQUFWLlyJV5++WUkJibiu+++q9Tm119/RceOHeHr64uwsDA89dRTACqm/i5duoQJEyZAoVBAoVAAAKZPn442bdroHGPevHmIiorS/H7o0CE8/PDDCAsLg0qlQo8ePXD06FHJ/X7sscdQq1atSv0tKCjA6tWrMXLkSGRnZ2Pw4MGoW7cu/P390bJlSyxfvtzkcQ1NmQYHB+uc58qVK3j22WcRHByM0NBQPPnkk7h48aLm/h07dqBTp04ICAhAcHAwunbtikuXLkn+24hIHgZlRFQlrFq1Cs2bN0ezZs0wdOhQfPvtt9De2nfjxo146qmn0LdvXxw7dgxbt25Fp06dAABr1qxBvXr18N577yEzMxOZmZmSz3vnzh0MHz4ce/bswf79+9G0aVP07dsXd+7ckfR4T09PDBs2DN99951Of1evXo3y8nIMHjwYRUVFaN++PTZu3IjU1FSMHj0azz//PA4ePCi5n/pKS0uRkJCAwMBA7N69G3v37kWNGjXQp08flJSUoKysDP369UOPHj1w4sQJJCcnY/To0ZqAlYhsz9PZHSAisoVvvvkGQ4cOBQD06dMHeXl52LlzJ3r27AkA+OCDDzBo0CDMmDFD85jWrVsDAEJDQ+Hh4YHAwEBERETIOm+vXr10fv/yyy8RHByMnTt34rHHHpN0jBdffBFz5szR6e+SJUswYMAAqFQqqFQqvPHGG5r2r7zyCn7//XesWrVKE1jKtXLlSqjVanz99deaQGvJkiUIDg7Gjh070KFDB+Tl5eGxxx5D48aNAQAtWrSw6FxEJA1HyojI7Z09exYHDx7E4MGDAVSMPg0cOBDffPONpk1KSgp69+5t83Nfv34do0aNQtOmTaFSqRAUFISCggJcvnxZ8jGaN2+OBx54AN9++y0A4MKFC9i9ezdGjhwJACgvL8fMmTPRsmVLhIaGokaNGvj9999lnUPf8ePHceHCBQQGBqJGjRqoUaMGQkNDUVRUhPT0dISGhuKFF15AQkICHn/8ccyfP1/WCCIRyceRMiJye9988w3KyspQp04dzW2CIMDHxwcLFiyASqWCn5+f7OMqlUqdKUWgYtpP2/Dhw5GdnY358+ejYcOG8PHxQVxcHEpKSmSda+TIkXjllVewcOFCLFmyBI0bN0aPHj0AAHPmzMH8+fMxb948tGzZEgEBAXjttddMnkOhUJjse0FBAdq3b4+lS5dWemytWrUAVIycvfrqq0hKSsLKlSvxzjvvYPPmzejSpYusv42IpOFIGRG5tbKyMvzvf//DJ598gpSUFM3P8ePHUadOHU1CfKtWrbB161ajx/H29kZ5ebnObbVq1UJWVpZOcJOSkqLTZu/evXj11VfRt29f3H///fDx8cGtW7dk/x3PPvsslEolli1bhv/973948cUXNdOKe/fuxZNPPomhQ4eidevWaNSoEc6dO2fyeLVq1dIZ2Tp//jzu3r2r+b1du3Y4f/48wsPD0aRJE50flUqlade2bVtMmTIF+/btQ2xsLJYtWyb7byMiaRiUEZFb27BhA27fvo2RI0ciNjZW52fAgAGaKcxp06Zh+fLlmDZtGk6fPo2TJ0/i448/1hwnKioKu3btwtWrVzVBVc+ePXHz5k3Mnj0b6enpWLhwIX777Ted8zdt2hQ//PADTp8+jQMHDmDIkCEWjcrVqFEDAwcOxJQpU5CZmYkXXnhB5xybN2/Gvn37cPr0abz00ku4fv26yeP16tULCxYswLFjx3D48GGMGTMGXl5emvuHDBmCsLAwPPnkk9i9ezcyMjKwY8cOvPrqq/jrr7+QkZGBKVOmIDk5GZcuXcIff/yB8+fPM6+MyI4YlBGRW/vmm28QHx+vM7ojGjBgAA4fPowTJ06gZ8+eWL16NdavX482bdqgV69eOqsX33vvPVy8eBGNGzfWTN+1aNECn3/+ORYuXIjWrVvj4MGDOgn34vlv376Ndu3a4fnnn8err76K8PBwi/6WkSNH4vbt20hISNCZin3nnXfQrl07JCQkoGfPnoiIiEC/fv1MHuuTTz5B/fr18eCDD+K5557DG2+8AX9/f839/v7+2LVrFxo0aID+/fujRYsWGDlyJIqKihAUFAR/f3+cOXMGAwYMwH333YfRo0dj7NixeOmllyz624jIPIWgn3RARERERA7HkTIiIiIiF8CgjIiIiMgFMCgjIiIicgEMyoiIiIhcAIMyIiIiIhfAoIyIiIjIBTAoIyIiInIBDMqIiIiIXACDMiIiIiIXwKCMiIiIyAUwKCMiIiJyAQzKiIiIiFzA/wM/+77g8i57IgAAAABJRU5ErkJggg==\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "Note: Remember to adapt these steps based on the specific characteristics of your dataset and the requirements of your supervised learning task. Additionally, choose an appropriate machine learning algorithm depending on whether your task is regression, classification, etc. So far we evaluated regression. Now we will build learning classification model"
      ],
      "metadata": {
        "id": "NlBQI0OXiF9P"
      }
    }
  ]
}
