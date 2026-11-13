import { fireEvent, render, screen } from '@testing-library/react';
import App from './App';
import { searchSimilarImages } from './api';

jest.mock('./api', () => ({
  getImageSrc: (result) => result.image_url,
  searchSimilarImages: jest.fn(),
}));

test('renders the search interface and empty state', () => {
  render(<App />);

  expect(screen.getByRole('heading', { name: /image similarity search/i })).toBeInTheDocument();
  expect(screen.getByText(/no matches found/i)).toBeInTheDocument();
});

test('uploads a query and renders the backend relative image URL', async () => {
  searchSimilarImages.mockResolvedValue({
    results: [{
      id: 12,
      score: 0.9876,
      path: 'nested dir/café photo.jpg',
      image_url: '/images/nested%20dir/caf%C3%A9%20photo.jpg',
    }],
  });
  render(<App />);

  const file = new File(['image-bytes'], 'query.jpg', { type: 'image/jpeg' });
  fireEvent.change(screen.getByLabelText(/upload image file/i), {
    target: { files: [file] },
  });
  fireEvent.click(screen.getByRole('button', { name: /search for similar images/i }));

  const image = await screen.findByRole('img', { name: 'Match 12' });
  expect(searchSimilarImages).toHaveBeenCalledWith(file);
  expect(image).toHaveAttribute(
    'src',
    '/images/nested%20dir/caf%C3%A9%20photo.jpg'
  );
  expect(screen.getByText('Score: 0.988')).toBeInTheDocument();
});
