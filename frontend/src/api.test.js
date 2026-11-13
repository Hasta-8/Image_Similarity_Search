import {
  buildApiUrl,
  getApiBaseUrl,
  getImageSrc,
  getThumbnailSrc,
  searchSimilarImages,
} from './api';

test('builds same-origin API paths by default', () => {
  expect(buildApiUrl('/search-image', '')).toBe('/search-image');
  expect(buildApiUrl('ready', '')).toBe('/ready');
});

test('preserves an explicitly configured API prefix without duplicate slashes', () => {
  expect(buildApiUrl('/search-image', 'https://api.example.test/v1/'))
    .toBe('https://api.example.test/v1/search-image');
});

test('reads and normalizes the CRA API base URL setting', () => {
  const base = getApiBaseUrl({ REACT_APP_API_BASE_URL: 'https://api.example.test/v1/' });
  expect(base).toBe('https://api.example.test/v1');
  expect(buildApiUrl('/search-image', base)).toBe(
    'https://api.example.test/v1/search-image'
  );
});

test('sends multipart searches through the configured relative API path', async () => {
  const originalFetch = global.fetch;
  global.fetch = jest.fn().mockResolvedValue({
    ok: true,
    json: async () => ({ results: [] }),
  });
  const file = new File(['query'], 'query.jpg', { type: 'image/jpeg' });

  try {
    await searchSimilarImages(file);

    expect(global.fetch).toHaveBeenCalledWith('/search-image', {
      method: 'POST',
      body: expect.any(FormData),
    });
    expect(global.fetch.mock.calls[0][1].body.get('file')).toBe(file);
  } finally {
    global.fetch = originalFetch;
  }
});

test('keeps backend relative image URLs same-origin and already encoded', () => {
  const url = '/images/nested%20dir/caf%C3%A9%20photo.jpg';
  expect(getImageSrc({ image_url: url }, '')).toBe(url);
});

test('resolves relative image URLs against an explicitly configured API origin', () => {
  expect(
    getImageSrc(
      { image_url: '/images/nested%20dir/caf%C3%A9%20photo.jpg' },
      'https://api.example.test'
    )
  ).toBe('https://api.example.test/images/nested%20dir/caf%C3%A9%20photo.jpg');
});

test('encodes raw fallback paths by segment while preserving nested directories', () => {
  expect(
    getImageSrc({ path: 'nested dir/café photo.jpg' }, '')
  ).toBe('/nested%20dir/caf%C3%A9%20photo.jpg');
});

test('prefers the backend thumbnail URL and keeps it same-origin', () => {
  const result = {
    image_url: '/images/nested%20dir/caf%C3%A9.jpg',
    thumbnail_url: '/thumbnails/nested%20dir/caf%C3%A9.jpg',
  };
  expect(getThumbnailSrc(result, '')).toBe('/thumbnails/nested%20dir/caf%C3%A9.jpg');
  expect(getThumbnailSrc(result, 'https://api.example.com')).toBe(
    'https://api.example.com/thumbnails/nested%20dir/caf%C3%A9.jpg'
  );
});

test('falls back to the full image when no thumbnail URL is provided', () => {
  expect(getThumbnailSrc({ image_url: '/images/a.jpg' }, '')).toBe('/images/a.jpg');
  expect(getThumbnailSrc({ image_url: '/images/a.jpg', thumbnail_url: null }, '')).toBe(
    '/images/a.jpg'
  );
  expect(getThumbnailSrc(null, '')).toBeNull();
});
