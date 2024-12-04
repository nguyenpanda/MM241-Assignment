import numpy as np

from policy import Policy


class Policy2250013(Policy):
    def __init__(self):
        super().__init__()

    def get_action(self, observation, info):
        # observation['stocks']: tuple[np.ndarray]
        # observation['products']: tuple[dict[np.ndarray, int]]
        stocks = observation['stocks']
        products = observation['products']

        sorted_products = sorted(
            products,
            key=lambda _p: _p['size'][0] * _p['size'][1],
            reverse=True
        )

        for product in sorted_products:
            product_size = product['size']

            if product['quantity'] > 0:
                for stock_idx, stock in enumerate(stocks):
                    stock_matrix = np.array(stock)

                    best_fit = None

                    for x in range(stock_matrix.shape[0] - product_size[0] + 1):
                        for y in range(stock_matrix.shape[1] - product_size[1] + 1):
                            candidate_area = stock_matrix[
                                x:x + product_size[0],
                                y:y + product_size[1],
                            ]

                            if np.all(candidate_area == -1):
                                waste = stock_matrix.size - product_size[0] * product_size[1]
                                if best_fit is None or waste < best_fit['waste']:
                                    best_fit = {
                                        'x': x,
                                        'y': y,
                                        'waste': waste,
                                    }

                    if best_fit:
                        return {
                            'stock_idx': stock_idx,
                            'size': product_size,
                            'position': (best_fit['x'], best_fit['y']),
                        }

        return None
