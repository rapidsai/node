import { DataFrame } from '@rapidsai/cudf';

export default function handler(req, res) {
  const { param } = req.query
  res.end(`Post: ${param.join(', ')}`)
}
