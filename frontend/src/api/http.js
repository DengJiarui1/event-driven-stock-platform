import axios from 'axios'

const http = axios.create({
  baseURL: '/',
  timeout: 10000
})

http.interceptors.response.use(
  (response) => response,
  (error) => {
    console.error('接口请求失败：', error)
    return Promise.reject(error)
  }
)

export default http