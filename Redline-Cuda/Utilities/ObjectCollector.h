#pragma once
#include <memory>
#include <vector>
#include <unordered_map>

using namespace std;

namespace Redline
{
	template<typename T>
	class ObjectCollector
	{
	private:	
		unordered_map<T*, int> _pointerToIndex;
		vector<T*> _values;

	public:

		void AddItem(T* item);

		void AddItem(shared_ptr<T> item);

		int GetIndexForItem(T* item);

		int GetIndexForItem(shared_ptr<T> item);

		vector<T> GetCopyOfValues();
	};

	template<typename T>
	inline void ObjectCollector<T>::AddItem(T* item)
	{
		if (_pointerToIndex.find(item) == _pointerToIndex.end())
		{
			int newIndex = _values.size();
			_pointerToIndex[item] = newIndex;
			_values.push_back(*item);
		}
	}

	template<typename T>
	inline void ObjectCollector<T>::AddItem(shared_ptr<T> item)
	{
		AddItem(item.get());
	}
	template<typename T>
	inline int ObjectCollector<T>::GetIndexForItem(T* item)
	{
		return 0;
	}
	template<typename T>
	inline int ObjectCollector<T>::GetIndexForItem(shared_ptr<T> item)
	{
		return GetIndexForItem(item.get());
	}

	template<typename T>
	inline vector<T> ObjectCollector<T>::GetCopyOfValues()
	{
		return vector<T>(_values);
	}
}