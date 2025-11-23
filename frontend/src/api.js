import axios from 'axios';

const API_URL = 'http://127.0.0.1:5000';

export const predictAnemia = async (data) => {
    try {
        const response = await axios.post(`${API_URL}/predict`, data);
        return response.data;
    } catch (error) {
        console.error("Error predicting anemia:", error);
        throw error;
    }
};

export const checkHealth = async () => {
    try {
        const response = await axios.get(`${API_URL}/health`);
        return response.data;
    } catch (error) {
        console.error("Health check failed:", error);
        throw error;
    }
};
