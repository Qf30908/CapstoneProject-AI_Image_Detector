import axios from "axios";

const API_URL = "http://127.0.0.1:5000";

export const predictImage = async (image) => {
    const formData = new FormData();
    formData.append("image", image);

    const response = await axios.post(
        `${API_URL}/predict`,
        formData
    );

    return response.data;
};