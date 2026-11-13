import { fireEvent, render, screen } from '@testing-library/react';
import App from './App';
import { searchSimilarImages } from './api';

jest.mock('./api', () => ({
  getImageSrc: (result) => result.image_url,
  getThumbnailSrc: (result) => result.thumbnail_url || result.image_url,
  searchSimilarImages: jest.fn(),
}));

test('renders the search interface and empty state', () => {
  render(<App />);

  expect(screen.getByRole('heading', { name: /image similarity search/i })).toBeInTheDocument();
  expect(screen.getByText(/no matches found/i)).toBeInTheDocument();
});

test('uploads a query and renders the thumbnail linked to the full image', async () => {
  searchSimilarImages.mockResolvedValue({
    results: [{
      id: 12,
      score: 0.9876,
      path: 'nested dir/café photo.jpg',
      image_url: '/images/nested%20dir/caf%C3%A9%20photo.jpg',
      thumbnail_url: '/thumbnails/nested%20dir/caf%C3%A9%20photo.jpg',
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
    '/thumbnails/nested%20dir/caf%C3%A9%20photo.jpg'
  );
  expect(
    screen.getByRole('link', { name: 'Open full-size image for match 12' })
  ).toHaveAttribute('href', '/images/nested%20dir/caf%C3%A9%20photo.jpg');
  expect(screen.getByText('Score: 0.988')).toBeInTheDocument();
});

test('falls back to the full image once if the thumbnail fails to load', async () => {
  searchSimilarImages.mockResolvedValue({
    results: [{
      id: 7,
      score: 0.5,
      path: 'a.jpg',
      image_url: '/images/a.jpg',
      thumbnail_url: '/thumbnails/a.jpg',
    }],
  });
  render(<App />);

  const file = new File(['image-bytes'], 'query.jpg', { type: 'image/jpeg' });
  fireEvent.change(screen.getByLabelText(/upload image file/i), {
    target: { files: [file] },
  });
  fireEvent.click(screen.getByRole('button', { name: /search for similar images/i }));

  const image = await screen.findByRole('img', { name: 'Match 7' });
  fireEvent.error(image);
  expect(image).toHaveAttribute('src', '/images/a.jpg');

  fireEvent.error(image);
  expect(image).toHaveAttribute('src', '/images/a.jpg');
});
