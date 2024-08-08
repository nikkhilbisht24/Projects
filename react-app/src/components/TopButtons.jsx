const TopButtons = ({ setQuery }) => {
    const cities = [
      {
        id: 1,
        title: 'London'
      },
      {
        id: 2,
        title: 'Sydney'
      },
      {
        id: 3,
        title: 'Tokyo'
      },
      {
        id: 4,
        title: 'Toronto'
      },
      {
        id: 5,
        title: 'Paris'
      }
    ];
    return (
      <div className='flex items-center justify-around my-6'>
        {cities.map((city) => (
          <button key={city.id} className="text-lg font-medium hover:bg-gray-700/20 px-3 py-2 rounded-md transition ease-in" onClick={() => setQuery({ q: city.title })}>
            {city.title}
          </button>
        ))}
      </div>
    );
  };
  
  export default TopButtons;
  