// Copyright (c) 2021, NVIDIA CORPORATION.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.




// Need to bring these over as well
// new class -> rapidsjsdom
//    extend the base jsdom class, provide a function that automatically creates the localhost URL
//    add a before parse thing that will attatch everything to the window
// First test:
//    construct an instance of JSDOM
//    can require file
//
//

// imageloader class needs to be moved over
//


//import { JSDOM, ResourceLoader } from 'jsdom';
import * as jsdom from 'jsdom';


export class RapidsJSDOM extends jsdom.JSDOM{

}
